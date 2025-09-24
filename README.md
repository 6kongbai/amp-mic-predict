# Bacterial Genomic Sequence Datasets for Machine Learning

This repository contains genomic sequence data for three clinically relevant bacterial species. The data is pre-split into training and testing sets, making it suitable for developing and evaluating machine learning models for tasks such as species classification, antimicrobial resistance (AMR) gene prediction, or other bioinformatics analyses.

## 🧬 Dataset Description

The dataset is organized by species, each identified by its scientific name and ATCC (American Type Culture Collection) strain number. The ATCC is a non-profit organization that collects, stores, and distributes standard reference microorganisms for research and development.

The three species included are:

-   ***Escherichia coli*** **ATCC 25922**: A well-studied, gram-negative bacterium commonly found in the lower intestine of warm-blooded organisms.
-   ***Pseudomonas aeruginosa*** **ATCC 27853**: A common gram-negative, rod-shaped bacterium that can cause disease in plants, animals, and humans.
-   ***Staphylococcus aureus*** **ATCC 25923**: A gram-positive, round-shaped bacterium that is a common member of the body's microbiota but can also be an opportunistic pathogen.

## 📁 Directory Structure

The data is organized into a clear, hierarchical structure. Each species has its own directory containing two FASTA files: one for training and one for testing.

```
data/
├── Escherichia coli ATCC 25922/
│   ├── train.fasta
│   └── test.fasta
│
├── Pseudomonas aeruginosa ATCC 27853/
│   ├── train.fasta
│   └── test.fasta
│
└── Staphylococcus aureus ATCC 25923/
    ├── train.fasta
    └── test.fasta
```

-   **`train.fasta`**: Contains the sequences intended for training your model.
-   **`test.fasta`**: Contains the sequences intended for evaluating the performance of your trained model.




