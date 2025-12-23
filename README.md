# Neural Speech Decoding with Modified Greedy Decoding, 3-Gram Shallow Fusion, and Data Augmentation

This repository contains experiments on improving a baseline GRU-based neural speech decoder by incorporating **external language model information** and **neural data augmentations**. The project evaluates a **modified greedy decoding strategy with shallow fusion of a 3-gram phoneme language model**, alongside several biologically motivated data augmentation techniques.

This work was developed for **Neural Signal Processing (ECE C143 / C243A)** at UCLA.

---

## Overview

The baseline model maps neural features—**threshold crossings (TX)** and **spike band power (SBP)**—to phoneme sequences using a GRU trained with **Connectionist Temporal Classification (CTC) loss**. The baseline model achieves approximately **23% character error rate (CER)**.

This project explores two main approaches to improving performance:

1. **Language-aware decoding**
   - Construction of a phoneme-level **3-gram language model**
   - **Modified greedy decoding** with shallow fusion of CTC logits and LM scores
   - Aggressive LM-based post-decoding autocorrection (evaluated but discarded)

2. **Neural data augmentation**
   - Rolling Z-score normalization
   - Poisson perturbation of spike counts
   - Electrode-level feature masking
   - Time masking, time-feature masking, and time shifting
   - PCA analysis (evaluated but not adopted)

---

## Key Contributions

### **3-Gram Phoneme Language Model**
- Trained from a phoneme dictionary (CMUdict)
- Models conditional probabilities of a phoneme given the previous two
- Produces **log-scores**, not full sequence probabilities
- Used strictly as a scoring function during decoding

### **Modified Greedy Decoding (Shallow Fusion)**
At each timestep:
1. Select the top-K phonemes by CTC logit
2. Compute LM scores for each candidate
3. Combine scores using a weighted sum
4. Choose the phoneme with the highest fused score

No beam search is used — decoding remains greedy.

### **Data Augmentation Suite**
- Augmentations are biologically motivated and modular
- Designed to model electrode failure, temporal jitter, and neural variability
- Evaluated independently and in combination

---

## Results Summary

| Method | Final CER |
|------|-----------|
| Baseline GRU | ~23% |
| 3-Gram Shallow Fusion (no tuning) | ~33% |
| Rolling Z-Score (window = 2) | ~23.6% |
| Time / Feature / Shift Masking | ~23–24% |
| Feature Masking + Poisson Noise | **21.88%** |

**Key observations:**
- Shallow fusion requires extensive hyperparameter tuning to be effective
- Most time-domain augmentations had minimal impact
- Electrode-level feature masking combined with Poisson noise produced the best result

---

