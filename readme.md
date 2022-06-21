# Improving classification by fine-grained labels uncertainty-driven estimation

# Introduction
This repository contains a PyTorch implementation of a deep learning based image-classification pipeline. Our proposal advocates that, in a coarse-grained label classification context, the fine-tuning of a pre-trained model on fine-grained label classification (defined via clustering techniques) shows better performance than training the classifier directly on coarse-grained labels. We focus on demonstrating and analyzing how, by obtaining the labels of the 'fine-grained' classes, it is possible to improve the accuracy of a CNN in the classification of coarse-grained classes.

# Overview

In the first step, feature embeddings are extracted through a CNN  trained on solving a classification task on coarse labels (macro-class). In contrast with representation learning approaches that require clustering of K-means after learning the representations of the features as DeepCluster, we propose to 
- i) select the most representative images of each coarse class by calculating the uncertainty from its augmentations; 
- ii) apply a size reduction via UMAP; 
- iii)search fine-grained labels through a density-based clustering method as HDBSCAN. Empirically, it is possible to note how this combination of tools allows defining the possible fine-grained labels for each coarse class while keeping the semantics intact as much as possible. 

In a second step, the pre-trained model over fine-grained labels is fine-tuned on coarse-grained labels. Alternatively, it is possible to carry out a single net training using a multi-head approach: one head of the net is used to calculate the loss on the fine-grained classes while the other on the coarse-grained classes.

# Pipeline
## 1. Step 1

### (1.1) Training coarse-label classifier  [ResNet18]

### (1.2) Features Extraction

### (1.3) Uncertainty Filter

### (1.4) UMAP + HBSCAN

## 2. Step 2

### (1.1) Fine-tuning Strategy (FN)

### (1.2) Multi-Head Strategy (MH)

# Requirements
Major dependencies are:

