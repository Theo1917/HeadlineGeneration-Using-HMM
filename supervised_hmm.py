#!/usr/bin/env python3
"""
HMM-Based Headline Generator
Uses Hidden Markov Models to learn headline generation patterns
"""

import argparse
import json
import os
import re
import random
import math
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import numpy as np


@dataclass
class Article:
    """Data class for article information"""
    title: str
    text: str
    summary: str
    coverage: float


@dataclass
class RougeScores:
    """Data class for ROUGE evaluation scores"""
    rouge1: float
    rouge2: float
    rougeL: float

    def __str__(self) -> str:
        return f"R1: {self.rouge1:.3f}, R2: {self.rouge2:.3f}, RL: {self.rougeL:.3f}"


class POSTagger:
    """Simple rule-based POS tagger for state assignment"""
    
    # Common word categories
    DETERMINERS = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
    PREPOSITIONS = {'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'about', 'as', 'into', 'after', 'amid'}
    CONJUNCTIONS = {'and', 'or', 'but', 'nor'}
    PRONOUNS = {'he', 'she', 'it', 'they', 'we', 'i', 'you', 'who', 'what'}
    AUXILIARIES = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'will', 'would', 'could', 'should'}
    
    # Common verbs
    VERBS = {
        'says', 'announces', 'reveals', 'confirms', 'reports', 'launches', 'unveils',
        'faces', 'wins', 'loses', 'seeks', 'plans', 'calls', 'urges', 'warns',
        'approves', 'rejects', 'begins', 'ends', 'starts', 'makes', 'takes', 'gives'
    }
    
    @classmethod
    def tag_word(cls, word: str, position: int, total: int) -> str:
        """Assign a POS tag to a word based on simple rules"""
        word_lower = word.lower()
        
        # Check if it's a special position
        if position == 0:
            # First word is often a noun or proper noun
            if word[0].isupper() and len(word) > 1:
                return 'NNP'  # Proper noun
            return 'NN'  # Common noun
        
        # Check word categories
        if word_lower in cls.DETERMINERS:
            return 'DT'
        if word_lower in cls.PREPOSITIONS:
            return 'IN'
        if word_lower in cls.CONJUNCTIONS:
            return 'CC'
        if word_lower in cls.PRONOUNS:
            return 'PRP'
        if word_lower in cls.AUXILIARIES:
            return 'VB'
        if word_lower in cls.VERBS or word_lower.endswith(('s', 'ed', 'ing')):
            return 'VB'
        
        # Check for proper nouns (capitalized)
        if word[0].isupper() and len(word) > 1:
            return 'NNP'
        
        # Check for adjectives (common endings)
        if word_lower.endswith(('ful', 'less', 'ous', 'ive', 'al')):
            return 'JJ'
        
        # Check for numbers
        if word.isdigit():
            return 'CD'
        
        # Default to noun
        return 'NN'

    @classmethod
    def tag_sequence(cls, words: List[str]) -> List[Tuple[str, str]]:
        """Tag a sequence of words"""
        tagged = []
        total = len(words)
        for i, word in enumerate(words):
            tag = cls.tag_word(word, i, total)
            tagged.append((word, tag))
        return tagged


class HiddenMarkovModel:
    """
    HMM for headline generation
    States: POS tags (NN, VB, NNP, etc.)
    Observations: Words
    """
    
    def __init__(self, smoothing: float = 1e-10):
        self.smoothing = smoothing
        
        # Transition probabilities: P(state_t | state_t-1)
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = defaultdict(dict)
        
        # Emission probabilities: P(word | state)
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.emission_probs = defaultdict(dict)
        
        # Initial state probabilities: P(state_0)
        self.initial_counts = defaultdict(int)
        self.initial_probs = {}
        
        # Bigram model for word sequences
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.bigram_probs = defaultdict(dict)
        
        # Vocabulary
        self.states = set()
        self.vocabulary = set()
        
        # Special tokens
        self.START_STATE = '<START>'
        self.END_STATE = '<END>'
        self.START_WORD = '<START_WORD>'

    def train(self, headlines: List[str]) -> None:
        """Train HMM on headline sequences"""
        print(f"Training HMM on {len(headlines)} headlines...")
        
        tagger = POSTagger()
        
        for headline in headlines:
            words = headline.strip().split()
            if not words:
                continue
            
            # Tag the sequence
            tagged = tagger.tag_sequence(words)
            
            # Extract states and observations
            states = [tag for _, tag in tagged]
            observations = [word.lower() for word, _ in tagged]
            
            # Update initial state counts
            if states:
                self.initial_counts[states[0]] += 1
                self.states.add(states[0])
            
            # Update transition and emission counts
            prev_state = self.START_STATE
            prev_word = self.START_WORD
            for i, (obs, state) in enumerate(zip(observations, states)):
                # Transition
                self.transition_counts[prev_state][state] += 1
                
                # Emission
                self.emission_counts[state][obs] += 1
                
                # Bigram (word-to-word)
                self.bigram_counts[prev_word][obs] += 1
                
                self.states.add(state)
                self.vocabulary.add(obs)
                prev_state = state
                prev_word = obs
            
            # Final transition to END
            if states:
                self.transition_counts[prev_state][self.END_STATE] += 1
        
        # Convert counts to probabilities
        self._compute_probabilities()
        
        print(f"✓ Training complete")
        print(f"  States: {len(self.states)}")
        print(f"  Vocabulary: {len(self.vocabulary)}")
        print(f"  Bigrams: {sum(len(v) for v in self.bigram_counts.values())}")

    def _compute_probabilities(self) -> None:
        """Convert counts to log probabilities with smoothing"""
        # Initial probabilities
        total_initial = sum(self.initial_counts.values())
        for state in self.states:
            count = self.initial_counts.get(state, 0)
            prob = (count + self.smoothing) / (total_initial + self.smoothing * len(self.states))
            self.initial_probs[state] = math.log(prob)
        
        # Transition probabilities
        all_states = self.states | {self.START_STATE, self.END_STATE}
        for prev_state in all_states:
            total = sum(self.transition_counts[prev_state].values())
            if total == 0:
                continue
            
            for state in all_states:
                count = self.transition_counts[prev_state].get(state, 0)
                prob = (count + self.smoothing) / (total + self.smoothing * len(all_states))
                self.transition_probs[prev_state][state] = math.log(prob)
        
        # Emission probabilities
        for state in self.states:
            total = sum(self.emission_counts[state].values())
            if total == 0:
                continue
            
            for word in self.vocabulary:
                count = self.emission_counts[state].get(word, 0)
                prob = (count + self.smoothing) / (total + self.smoothing * len(self.vocabulary))
                self.emission_probs[state][word] = math.log(prob)
        
        # Bigram probabilities
        vocab_with_start = self.vocabulary | {self.START_WORD}
        for prev_word in vocab_with_start:
            total = sum(self.bigram_counts[prev_word].values())
            if total == 0:
                continue
            
            for word in self.vocabulary:
                count = self.bigram_counts[prev_word].get(word, 0)
                prob = (count + self.smoothing) / (total + self.smoothing * len(self.vocabulary))
                self.bigram_probs[prev_word][word] = math.log(prob)
    
    def get_bigram_score(self, prev_word: str, word: str) -> float:
        """Get bigram probability score for word given previous word"""
        if prev_word not in self.bigram_probs:
            return math.log(self.smoothing)
        return self.bigram_probs[prev_word].get(word, math.log(self.smoothing))

    def get_top_words_for_state(self, state: str, n: int = 20) -> List[Tuple[str, int]]:
        """Get most likely words for a given state"""
        if state not in self.emission_counts:
            return []
        
        word_counts = self.emission_counts[state]
        return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_likely_next_states(self, current_state: str, n: int = 5) -> List[str]:
        """Get most likely next states"""
        if current_state not in self.transition_counts:
            return list(self.states)[:n]
        
        transitions = self.transition_counts[current_state]
        sorted_states = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        return [state for state, _ in sorted_states[:n] if state != self.END_STATE]

    def viterbi_decode(self, observations: List[str]) -> List[str]:
        """
        Find most likely state sequence for given observations using Viterbi algorithm
        """
        if not observations:
            return []
        
        T = len(observations)
        states_list = list(self.states)
        
        # Initialize Viterbi matrix and backpointer matrix
        viterbi = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]
        
        # Initialization step
        for state in states_list:
            initial_prob = self.initial_probs.get(state, math.log(self.smoothing))
            emission_prob = self.emission_probs.get(state, {}).get(
                observations[0], math.log(self.smoothing)
            )
            viterbi[0][state] = initial_prob + emission_prob
            backpointer[0][state] = None
        
        # Recursion step
        for t in range(1, T):
            for state in states_list:
                max_prob = float('-inf')
                max_state = None
                
                for prev_state in states_list:
                    trans_prob = self.transition_probs.get(prev_state, {}).get(
                        state, math.log(self.smoothing)
                    )
                    prob = viterbi[t-1][prev_state] + trans_prob
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_state
                
                emission_prob = self.emission_probs.get(state, {}).get(
                    observations[t], math.log(self.smoothing)
                )
                viterbi[t][state] = max_prob + emission_prob
                backpointer[t][state] = max_state
        
        # Termination step
        max_prob = float('-inf')
        best_final_state = None
        for state in states_list:
            if viterbi[T-1][state] > max_prob:
                max_prob = viterbi[T-1][state]
                best_final_state = state
        
        # Backtrack to find best path
        best_path = [best_final_state]
        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])
        
        return best_path


class HeadlineGenerator:
    """HMM-based headline generator"""
    
    def __init__(self):
        self.hmm = HiddenMarkovModel()
        self.tagger = POSTagger()
        self.trained = False
        
        # Store article keywords for content selection
        self.article_keywords = []

    def train(self, articles: List[Article]) -> None:
        """Train the HMM on headlines"""
        headlines = [article.title for article in articles if article.title]
        
        # Clean headlines - remove source markers
        cleaned_headlines = []
        for headline in headlines:
            # Remove common source markers
            cleaned = re.sub(r'\s*[-:]\s*(People\.com|Nytimes\.com|CNN\.com|BBC\.com|Reuters|AP|AFP)\s*$', '', headline, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*\|\s*(People\.com|Nytimes\.com|CNN|BBC|Reuters)\s*$', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned:
                cleaned_headlines.append(cleaned)
        
        self.hmm.train(cleaned_headlines)
        self.trained = True

    def extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        """Extract important keywords from article using advanced TF-IDF-like scoring"""
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Expanded stop words
        stop_words = {
            'the', 'and', 'that', 'this', 'with', 'from', 'have', 'been',
            'were', 'said', 'their', 'what', 'when', 'which', 'about', 'will',
            'there', 'them', 'would', 'make', 'than', 'more', 'some', 'could',
            'other', 'into', 'very', 'after', 'also', 'just', 'where', 'most',
            'only', 'such', 'over', 'should', 'our', 'those', 'these', 'then',
            'who', 'has', 'had', 'but', 'not', 'can', 'all', 'one', 'two',
            'may', 'now', 'like', 'get', 'got', 'his', 'her', 'its', 'see',
            'did', 'does', 'don', 'doesn', 'wasn', 'weren', 'won', 'first',
            'made', 'way', 'many', 'much', 'well', 'even', 'back', 'use'
        }
        
        # Filter words
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        word_freq = Counter(filtered)
        
        # Get named entities (capitalized words in original text)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entity_words = []
        for e in entities:
            if e not in {'The', 'This', 'That', 'These', 'Those', 'There', 'They', 'When', 'Where', 'What', 'Which', 'Who'}:
                entity_words.extend(e.lower().split())
        
        # Mega boost for entity words (they're most important)
        for word in entity_words:
            if word in word_freq:
                word_freq[word] *= 6  # 6x weight for named entities
        
        # Extra boost for words in first sentence
        first_sentence = text[:150].lower()
        first_sent_words = re.findall(r'\b[a-zA-Z]{3,}\b', first_sentence)
        for word in first_sent_words:
            if word in word_freq and word not in stop_words:
                word_freq[word] = int(word_freq[word] * 2.0)  # Double boost for first sentence
        
        # Boost rare but meaningful words (appear 2-4 times - likely important specifics)
        for word, count in word_freq.items():
            if 2 <= count <= 4:
                word_freq[word] = int(word_freq[word] * 1.3)
        
        # Get top keywords by frequency
        top_words = [word for word, _ in word_freq.most_common(top_n * 3)]
        
        # Filter out words that are too common
        if self.trained:
            # Remove generic/common words that appear frequently in headlines
            blacklist = {'people', 'new', 'york', 'nytimes', 'com', 'news', 'says', 
                        'report', 'reports', 'reuters', 'according', 'times', 'told',
                        'year', 'years', 'day', 'week', 'time', 'video', 'photo',
                        'home', 'world', 'us', 'via', 'watch', 'read', 'look'}
            top_words = [w for w in top_words if w not in blacklist]
        
        return top_words[:top_n]

    def generate_state_sequence(self, length: int = 6) -> List[str]:
        """Generate a likely state sequence using the HMM"""
        if not self.trained:
            return ['NNP', 'VB', 'NN'] * (length // 3 + 1)
        
        states = []
        current_state = self.hmm.START_STATE
        
        for _ in range(length):
            # Get likely next states
            next_states = self.hmm.get_likely_next_states(current_state, n=5)
            
            if not next_states:
                break
            
            # Sample next state based on transition probabilities
            next_state = self._sample_next_state(current_state, next_states)
            states.append(next_state)
            current_state = next_state
        
        return states

    def _sample_next_state(self, current_state: str, candidates: List[str]) -> str:
        """Sample next state from candidates based on transition probabilities"""
        if not candidates:
            return random.choice(list(self.hmm.states))
        
        # Get transition probabilities (in log space)
        log_probs = []
        for state in candidates:
            log_prob = self.hmm.transition_probs.get(current_state, {}).get(
                state, math.log(self.hmm.smoothing)
            )
            log_probs.append(log_prob)
        
        # Convert from log space to probabilities
        max_log = max(log_probs)
        probs = [math.exp(lp - max_log) for lp in log_probs]
        
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # Sample
        return random.choices(candidates, weights=probs)[0]

    def select_word_for_state(self, state: str, keywords: List[str], 
                             used_words: Set[str], prev_word: str = None) -> Optional[str]:
        """Select best word for a given state from available keywords with advanced scoring"""
        # Expanded blacklist of words to avoid
        blacklist = {'people', 'new', 'york', 'nytimes', 'com', 'the', 'of', 'to', 
                    'for', 'and', 'in', 'on', 'at', 'by', 'was', 'were', 'is', 'are',
                    'has', 'had', 'will', 'can', 'may', 'been', 'being', 'with',
                    'from', 'his', 'her', 'their', 'its', 'our', 'your', 'out',
                    'said', 'did', 'says', 'told', 'asked', 'put', 'get'}
        
        # Get top words for this state from HMM
        top_words = self.hmm.get_top_words_for_state(state, n=70)
        top_words_set = {word for word, _ in top_words if word not in blacklist}
        
        # Find keywords that match this state and haven't been used
        candidates = []
        for i, kw in enumerate(keywords):
            if kw in top_words_set and kw not in used_words:
                # Score based on keyword position (earlier = MUCH more important)
                position_score = (len(keywords) - i) * 2
                
                # Get the count from HMM
                hmm_count = self.hmm.emission_counts[state].get(kw, 0)
                
                # Add bigram score if we have previous word (more weight)
                bigram_score = 0
                if prev_word:
                    bigram_score = math.exp(self.hmm.get_bigram_score(prev_word, kw)) * 150
                
                # Boost for words in top 5 keywords
                top5_bonus = 50 if i < 5 else 0
                
                # Combined score with enhanced weighting
                score = hmm_count * 1.5 + position_score * 20 + bigram_score + top5_bonus
                candidates.append((kw, score))
        
        # If we have matching candidates, select the best one
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        # Fallback: prefer keywords even if not in top words for this state
        unused_keywords = [kw for kw in keywords[:20] if kw not in used_words and kw not in blacklist]
        if unused_keywords:
            # If we have previous word, prefer words that form good bigrams
            if prev_word and len(unused_keywords) > 1:
                scored = []
                for kw in unused_keywords[:15]:  # Check top 15
                    # Combine bigram score with keyword position
                    bigram_score = math.exp(self.hmm.get_bigram_score(prev_word, kw))
                    kw_position = keywords.index(kw)
                    position_bonus = max(0, 20 - kw_position)
                    total_score = bigram_score * 10 + position_bonus
                    scored.append((kw, total_score))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[0][0]
            return unused_keywords[0]
        
        # Last resort: select most likely word for this state that hasn't been used
        for word, count in top_words:
            if word not in used_words and word not in blacklist:
                return word
        
        return None

    def generate(self, text: str, summary: str = "", 
                target_length: int = 6, max_len: int = 10) -> str:
        """Generate headline using advanced HMM with bigram smoothing and multi-sampling"""
        if not self.trained:
            return "Breaking News Today"
        
        # Extract keywords from both summary and text for better coverage
        keywords_summary = []
        keywords_text = []
        
        if summary and len(summary) > 50:
            keywords_summary = self.extract_keywords(summary, top_n=25)
        
        # Always get keywords from article text (first 800 chars for more context)
        text_start = text[:800] if len(text) > 800 else text
        keywords_text = self.extract_keywords(text_start, top_n=25)
        
        # Merge keywords, prioritizing summary keywords
        seen = set()
        keywords = []
        for kw in keywords_summary + keywords_text:
            if kw not in seen:
                keywords.append(kw)
                seen.add(kw)
                if len(keywords) >= 35:
                    break
        
        if not keywords:
            return "News Update"
        
        # Generate multiple candidates (5 attempts) and pick the best
        best_headline = None
        best_score = -1
        
        for attempt in range(5):  # Try 5 different generations
            # Vary target length slightly for diversity
            varied_length = target_length + random.randint(-1, 1)
            varied_length = max(4, min(8, varied_length))
            
            # Generate state sequence
            states = self.generate_state_sequence(varied_length)
            
            # Generate words for each state with bigram awareness
            headline_words = []
            used_words = set()
            prev_word = None
            
            for state in states:
                word = self.select_word_for_state(state, keywords, used_words, prev_word)
                if word:
                    headline_words.append(word)
                    used_words.add(word)
                    prev_word = word
                
                if len(headline_words) >= max_len:
                    break
            
            # Ensure minimum length with top keywords
            if len(headline_words) < 3:
                for kw in keywords[:5]:  # Use only top 5 keywords
                    if kw not in used_words:
                        headline_words.append(kw)
                        used_words.add(kw)
                        if len(headline_words) >= 3:
                            break
            
            # Advanced scoring for candidate quality
            if headline_words:
                # Heavily prefer headlines with top keywords
                keyword_score = sum(max(0, 40 - i * 1.5) for i, kw in enumerate(keywords[:20]) 
                                   if kw in headline_words)
                
                # Bonus for using keywords from both summary and text
                from_summary = sum(1 for w in headline_words if w in keywords_summary)
                from_text = sum(1 for w in headline_words if w in keywords_text)
                diversity_bonus = (from_summary * 5) + (from_text * 2)
                
                # Length bonus (prefer 5-7 words)
                ideal_length = 6
                length_penalty = abs(len(headline_words) - ideal_length) * 3
                
                # Total score
                score = keyword_score + diversity_bonus - length_penalty + len(headline_words) * 2
                
                if score > best_score:
                    best_score = score
                    best_headline = headline_words
        
        if not best_headline:
            return "News Update"
        
        # Capitalize properly
        headline = ' '.join(best_headline)
        return self._capitalize_headline(headline)

    @staticmethod
    def _capitalize_headline(headline: str) -> str:
        """Apply proper headline capitalization"""
        words = headline.split()
        if not words:
            return headline
        
        # Capitalize first word
        words[0] = words[0].capitalize()
        
        # Capitalize important words
        small_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 
                      'in', 'of', 'on', 'or', 'the', 'to'}
        
        for i in range(1, len(words)):
            if words[i].lower() not in small_words or i == len(words) - 1:
                words[i] = words[i].capitalize()
        
        return ' '.join(words)


class RougeEvaluator:
    """Evaluates headline quality using ROUGE metrics"""
    
    @staticmethod
    def get_ngrams(text: str, n: int) -> List[str]:
        """Extract n-grams from text"""
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    @staticmethod
    def calculate_lcs(seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

    @classmethod
    def evaluate(cls, reference: str, hypothesis: str) -> RougeScores:
        """Calculate ROUGE scores"""
        if not reference or not hypothesis:
            return RougeScores(0.0, 0.0, 0.0)
        
        # ROUGE-1 (unigram overlap)
        ref_unigrams = set(reference.lower().split())
        hyp_unigrams = set(hypothesis.lower().split())
        overlap_1 = len(ref_unigrams & hyp_unigrams)
        rouge1 = overlap_1 / len(ref_unigrams) if ref_unigrams else 0.0
        
        # ROUGE-2 (bigram overlap)
        ref_bigrams = set(cls.get_ngrams(reference, 2))
        hyp_bigrams = set(cls.get_ngrams(hypothesis, 2))
        overlap_2 = len(ref_bigrams & hyp_bigrams)
        rouge2 = overlap_2 / len(ref_bigrams) if ref_bigrams else 0.0
        
        # ROUGE-L (longest common subsequence)
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        lcs_length = cls.calculate_lcs(ref_words, hyp_words)
        rougeL = lcs_length / len(ref_words) if ref_words else 0.0
        
        return RougeScores(rouge1, rouge2, rougeL)


def load_articles(jsonl_path: str, max_samples: int = 100,
                 min_coverage: float = 0.8) -> List[Article]:
    """Load articles from JSONL file"""
    articles = []
    
    print(f"Loading articles from {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
                
                if data.get('coverage', 0) < min_coverage:
                    continue
                
                title = data.get('title', '').strip()
                text = data.get('text', '').strip()
                summary = data.get('summary', '').strip()
                coverage = data.get('coverage', 0.0)
                
                if title and text:
                    articles.append(Article(title, text, summary, coverage))
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Warning: Error loading line {i+1}: {e}")
                continue
    
    print(f"✓ Loaded {len(articles)} articles")
    return articles


def run_evaluation(articles: List[Article], train_ratio: float = 0.8,
                  num_test: int = 5, target_length: int = 6, 
                  max_len: int = 10) -> None:
    """Train HMM and evaluate headline generation"""
    if not articles:
        print("Error: No articles to evaluate!")
        return
    
    # Split into train and test
    random.shuffle(articles)
    split_idx = int(len(articles) * train_ratio)
    train_articles = articles[:split_idx]
    test_articles = articles[split_idx:]
    
    if not train_articles or not test_articles:
        print("Error: Not enough articles for train/test split!")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 HMM-Based Headline Generator")
    print(f"{'='*80}")
    print(f"Training set: {len(train_articles)} articles")
    print(f"Test set: {len(test_articles)} articles")
    print(f"{'='*80}\n")
    
    # Train the generator
    generator = HeadlineGenerator()
    generator.train(train_articles)
    
    # Test on all test articles or a sample
    evaluator = RougeEvaluator()
    
    # Use all test articles if num_test is -1, otherwise sample
    if num_test == -1:
        test_sample = test_articles
        print(f"\n{'='*80}")
        print(f"📰 Evaluating on ALL {len(test_sample)} test articles")
        print(f"{'='*80}\n")
    else:
        test_sample = random.sample(test_articles, min(num_test, len(test_articles)))
        print(f"\n{'='*80}")
        print(f"📰 Generation Results ({len(test_sample)} samples)")
        print(f"{'='*80}\n")
    
    total_scores = RougeScores(0.0, 0.0, 0.0)
    successful = 0
    
    for i, article in enumerate(test_sample, 1):
        try:
            # Generate headline
            predicted = generator.generate(article.text, article.summary, 
                                          target_length, max_len)
            
            # Evaluate
            scores = evaluator.evaluate(article.title, predicted)
            
            # Print detailed results only for first 10 and last 5 when evaluating all
            if num_test == -1:
                if i <= 10 or i > len(test_sample) - 5:
                    print(f"[{i}/{len(test_sample)}] Coverage: {article.coverage:.2f}")
                    print(f"  Reference: {article.title}")
                    print(f"  Generated: {predicted}")
                    print(f"  Scores: {scores}")
                    print(f"{'-'*80}\n")
                elif i == 11:
                    print(f"... Evaluating articles 11-{len(test_sample)-5} (not shown) ...\n")
            else:
                print(f"[Sample {i}] Coverage: {article.coverage:.2f}")
                print(f"  Reference: {article.title}")
                print(f"  Generated: {predicted}")
                print(f"  Scores: {scores}")
                print(f"{'-'*80}\n")
            
            total_scores.rouge1 += scores.rouge1
            total_scores.rouge2 += scores.rouge2
            total_scores.rougeL += scores.rougeL
            successful += 1
            
        except Exception as e:
            if num_test != -1 or i <= 10 or i > len(test_sample) - 5:
                print(f"[Sample {i}] Error: {e}\n{'-'*80}\n")
    
    # Print summary
    if successful > 0:
        avg_r1 = total_scores.rouge1 / successful
        avg_r2 = total_scores.rouge2 / successful
        avg_rL = total_scores.rougeL / successful
        
        print(f"{'='*80}")
        print(f"📊 Summary Statistics ({successful} samples)")
        print(f"{'='*80}")
        print(f"  Average ROUGE-1:  {avg_r1:.3f}")
        print(f"  Average ROUGE-2:  {avg_r2:.3f}")
        print(f"  Average ROUGE-L:  {avg_rL:.3f}")
        print(f"{'='*80}\n")
    
    print("✅ Evaluation complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="HMM-Based Headline Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--jsonl', required=True,
                       help='Path to JSONL file containing articles')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples to load')
    parser.add_argument('--test_samples', type=int, default=5,
                       help='Number of samples to test')
    parser.add_argument('--target_length', type=int, default=6,
                       help='Target headline length in words')
    parser.add_argument('--max_len', type=int, default=10,
                       help='Maximum headline length in words')
    parser.add_argument('--min_coverage', type=float, default=0.8,
                       help='Minimum coverage threshold for articles')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.jsonl):
        print(f"Error: File '{args.jsonl}' not found!")
        return 1
    
    try:
        # Load articles
        articles = load_articles(args.jsonl, args.max_samples, args.min_coverage)
        
        if len(articles) < 10:
            print("Error: Need at least 10 articles for training and testing!")
            return 1
        
        # Run evaluation
        run_evaluation(articles, args.train_ratio, args.test_samples,
                      args.target_length, args.max_len)
        
        return 0
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())