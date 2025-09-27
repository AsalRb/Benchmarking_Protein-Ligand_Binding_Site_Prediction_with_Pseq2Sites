
# Benchmarking Protein–Ligand Binding Site Prediction with Pseq2Sites

---

## Overview
This repository hosts my **Bachelor Project** on benchmarking machine learning approaches for **protein–ligand binding site prediction** using only **protein sequences**.

The project is based on [**Pseq2Sites**](https://github.com/Blue1993/Pseq2Sites), a state-of-the-art deep learning method that combines:
- **1D Convolutional Neural Networks (CNNs):** capture local sequence motifs (potential binding pockets).
- **Position-based Attention Mechanism:** model long-range dependencies between amino acids that may form binding sites together.

Unlike traditional **3D structure-based tools**, which require experimentally solved structures, Pseq2Sites uses only protein sequences, making it faster and more widely applicable in **drug discovery and bioinformatics**.

---

## Objectives
- Preprocess large-scale protein–ligand datasets (**scPDB** and **PDBbind**).
- Train and benchmark the **Pseq2Sites** model.
- Evaluate using standard benchmark datasets (**COACH420, HOLO4K, CSAR NRC-HiQ**).
- Analyze predictive power with multiple metrics (success rate, F-score, precision, recall, G-mean, accuracy).

---

## Workflow
1. **Dataset Preparation**  
   - Collect protein–ligand complexes from scPDB v2017 and PDBbind v2020.  
   - Apply filtering (sequence length ≤1500, ligand parsing checks, exclude peptide ligands).  
   - Prevent overlap between training and test data.  

2. **Feature Representation**  
   - Encode protein sequences using **ProtTrans embeddings** (protein language model).  

3. **Model Training & Benchmarking**  
   - Train **Pseq2Sites** with CNN + Attention modules.  
   - Hyperparameter tuning (CNN kernel sizes, dilation rates, attention heads, classification thresholds).  

4. **Evaluation**  
   - Benchmark against state-of-the-art methods (e.g., DeepCSeqSite, BiRDS, HoTS, P2Rank, DeepPocket).  
   - Use evaluation metrics: Success Rate (SR), Precision, Recall, F1/F2, G-mean, Accuracy.  

5. **Analysis & Visualization**  
   - Assess performance on **unseen proteins** (≤40% similarity to training).  
   - Visualize predicted binding sites compared to experimentally known binding pockets.  

---

## Current Progress 
- Starting **preprocessing pipeline** for scPDB and PDBbind.  
- Preparing datasets for input into **Pseq2Sites**.  
- Training and benchmarking stage (planned).  
- Evaluation, visualization, and analysis (planned).  

---

## Repository Structure
```

/Bachelor_Project_Pseq2Sites/
│── PDBbind_Preprocessing/         # Scripts for preprocessing PDBbind data
│── scPDB_Preprocessing/           # Scripts for preprocessing scPDB data
│── Pseq2Sites_presentation.pdf    # My project presentation slides
└── README.md                      # Project documentation (this file)

```

---

## Importance
- **Scientific Value:** Advances sequence-based binding site prediction, an area where accuracy has historically lagged behind structure-based methods.  
- **Drug Discovery:** Identifies potential druggable pockets without requiring expensive 3D structural data.  
- **Benchmarking Contribution:** Provides systematic performance comparisons to highlight strengths and limitations of Pseq2Sites.  
- **Practical Relevance:** Demonstrates how modern ML methods democratize protein–ligand interaction studies by removing reliance on structural databases.  

---

## References
- [Original Pseq2Sites GitHub Repository](https://github.com/Blue1993/Pseq2Sites)  
- Seo et al. (2024). *Pseq2Sites: Enhancing protein sequence-based ligand binding-site prediction accuracy via the deep convolutional network and attention mechanism.* **Engineering Applications of Artificial Intelligence, 127**, 107257. https://doi.org/10.1016/j.engappai.2023.107257  

