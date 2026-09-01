Emotion concepts replication
- Make a model desperate by making it fail multiple times on a test
- Measure the desperation through emotion concepts
- First - replicate part 1 by extracting the vectors and showing that they represent emotion in the model

Plan - Part 1
- Setup - 12 emotions (joy, sadness, anger, fear, disgust, surprise, calm, desperation, pride, shame, loneliness, excitement)
- Extraction - residual stream at every layer, pooled across tokens in the passage
- Vectors - per-emotion mean
- Tests (show that vectors are representative) - nearest vectors, intensity, implicit
- Bonus - find out which size of Qwen actually starts to have emotion concepts

Checklist
- [ ] Create dataset
- [ ] Extract & validate emotion concepts from a model
- [ ] Get model to cheat on a coding test