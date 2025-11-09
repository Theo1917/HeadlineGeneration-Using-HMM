# HeadlineGeneration-Using-HMM

Things required in the project ( basic understanding)
Hidden states (Z): latent topics or information importance levels (e.g., {main_event, supporting_fact, quote, background})
Observations (O): actual words or sentences in your article
Transition probabilities (A): how likely one topic leads to another
Emission probabilities (B): how likely each word (or sentence type) is emitted from a topic
Initial probabilities (π): how likely each topic is to start an article
Then, the headline and summary are derived from the most probable sequence  in both directions.
