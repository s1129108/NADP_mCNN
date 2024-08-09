# NADP_mCNN: Identifying NADP+-binding sites in chloroplast proteins using Protein Language Models and Multi-Window Convolutional Neural Networks

## Introduction:

Photosynthesis relies on proteins with NADP+ binding sites for converting light energy into chemical energy. NADP+ and its reduced form, NADPH, are essential for redox reactions in cellular metabolism and photosynthesis, particularly in the Calvin cycle. This study explores the critical role of NADP+ binding proteins in photosynthesis, especially under pathogen stress. Advanced computational models like ProtTrans, ESM2, ESM1b, and a multi-window scanning CNN model have shown high accuracy in predicting NADP+ binding sites. Integrating these predictions, the study enhances understanding of NADP+ binding dynamics, offering insights that could improve crop yields and sustainable energy solutions. This research connects computational predictions with experimental validation, advancing agricultural biotechnology and photosynthesis research.

## Graphical Abstract:

<img width="345" alt="image" src="https://github.com/user-attachments/assets/d7c696f7-e253-4c73-b9d9-ec8270ae6104">

## Dataset:

| Dataset | Protein sequence | NADP interacting residue | Non interacting residue |
|----------|:--------:|:---------:|:---------:|
| Train | 32 | 446 | 14140 |
| Test | 8 | 134 | 3403 |
| Total | 40 | 580 | 17543 |

## Quick Start:

### Step 1: Generate Data Features

Navigate to the data folder and utilize the FASTA file to produce additional data features, saving them in the dataset folder.

General usage:

```bash
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
````
Example usage:

```bash
python get_ProtTrans.py -in ./Train_seq/seq -out ./protTrans_output
````
### Step 2: Generate a complete dataset for training and testing

Using the embedded sequences to group them into numpy data fro sequences and labels

General usage:

```bash
python get_dataset.py -in "Your data feature Folder" -out "The destination folder of your output" -dt "Datatype of your feature" -w "Window Size" -label "Your data label Folder"  
````
Example usage:

```bash
python get_dataset.py -in ./protTrans_output -out ./protTrans_dataset -dt .prottrans -w 7 -label ./label
````

### Step 3: Execute prediction on complete dataset
Navigate to MCNN folder to execute the prediction on NADP+ binding site proteins.

Training usage:

```bash

python MCNN_NAD.py -d ProtTrans -n_dep 7 -ws 14 10 8 12 6 -n_feat 1280
````
Predition usage:

```bash

python MCNN_NAD.py -d ProtTrans -n_dep 7 -ws 14 10 8 12 6 -n_feat 1280 -vm independent
````

