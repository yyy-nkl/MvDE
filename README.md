# MvDE

MvDE: Integrating Multi-View Similarities and Deep Ensemble Learning for Metabolite–Neurodegenerative Disease Association Prediction.

## Overview

MvDE is a computational framework for metabolite–disease association (MDA) prediction.  
It integrates multi-view similarity information and heterogeneous feature learning modules to improve prediction performance on metabolite–disease association tasks.

The current implementation includes several core modules:

- **five_AE.py**: autoencoder-based feature extraction module
- **GAE.py**: graph autoencoder / graph attention autoencoder model definition
- **GAE_trainer.py**: training procedure for the GAE module
- **NMF.py**: non-negative matrix factorization-based feature extraction
- **main.py**: main entry point for running the full pipeline

## Framework

The MvDE framework mainly contains the following steps:

1. Construct metabolite and disease similarity representations
2. Learn low-dimensional features using:
   - autoencoder (AE)
   - graph autoencoder (GAE)
   - non-negative matrix factorization (NMF)
3. Concatenate multi-source features into integrated representations
4. Perform downstream metabolite–disease association prediction

## Data Access

Due to file size limitations, the full dataset is not directly included in this repository.  
Researchers who need the complete processed data may request it from the corresponding author by email.

