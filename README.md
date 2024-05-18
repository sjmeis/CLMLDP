# Collocation Extractor
Code for the anonymous PrivateNLP submission: *A Collocation-based Method for Addressing Challenges in Word-level Metric Differential Privacy*

## Getting Started
In this repository, you will find the following items:

- `CollocationExtractor.py`: class code for the GST and MST algorithms, as described in the paper
- `MLDP.py`: code for running MLDP mechanisms
- `data`: folder containing the data for bigram and trigram collocation extraction

In addition, we make our two trained embedding models public. Due to their large size, they must be downloaded at the following link: https://drive.google.com/drive/folders/1b_2QNSBBtmCuUAOLrQ2ZK41cKqHY-o-s?usp=sharing.

Note that for each embedding model, there are two necessary files. To load them, use `KeyedVectors` from the `gensim` package.