# Improving classification by fine-grained labels uncertainty-driven estimation

# Introduction
This repository contains a PyTorch implementation of a deep learning based image-classification pipeline. Our proposal advocates that, in a coarse-grained label classification context, the fine-tuning of a pre-trained model on fine-grained label classification (defined via clustering techniques) shows better performance than training the classifier directly on coarse-grained labels. We focus on demonstrating and analyzing how, by obtaining the labels of the 'fine-grained' classes, it is possible to improve the accuracy of a CNN in the classification of coarse-grained classes.

# Usage
## 1. PathMNIST tutorial


### (a) Tiling Patch 
```
python src/tile_WSI.py -s 512 -e 0 -j 32 -B 50 -M 20 -o <full_patch_to_output_folder> "full_path_to_input_slides/*/*.svs"
```
Mandatory parameters:


### (b) Training Patch Feature Extractor

```
python run.py
```

### (c) Constructing Graph
Go to './feature_extractor' and build graphs from patches:
```
python build_graphs.py --weights "path_to_pretrained_feature_extractor" --dataset "path_to_patches" --output "../graphs"
```

## 2. Training Graph-Transformer
Run the following script to train and store the model and logging files under "graph_transformer/saved_models" and "graph_transformer/runs".
```
bash scripts/train.sh
```
To evaluate the model. run
```bash scripts/test.sh```

Split training, validation, and testing dataset and store them in text files as:
```
sample1 \t label1
sample2 \t label2
LUAD/C3N-00293-23 \t luad
...
```

## 3. GraphCAM
To generate GraphCAM of the model on the WSI:
```
1. bash scripts/get_graphcam.sh
```
To visualize the GraphCAM:
```
2. bash scripts/vis_graphcam.sh
```
Note: Currently we only support generating GraphCAM for one WSI at each time.


GraphCAMs generated on WSIs across the runs performed via 5-fold cross validation are shown above. The same set of WSI regions are highlighted by our method across the various  cross-validation folds, thus indicating consistency of our technique in highlighting salient regions of interest. 

# Requirements
Major dependencies are:

