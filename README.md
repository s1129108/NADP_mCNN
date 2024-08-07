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

### Step 1: Generate features


