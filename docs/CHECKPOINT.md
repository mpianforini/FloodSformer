# Checkpoints
Download the appropriate checkpoint and save it a folder named `pretrain`. Please verify that the parameters `MODEL.PRETRAINED_AE` and `MODEL.PRETRAINED_VPTR` contain the correct paths to the AE and VPTR checkpoints (see files in [Parflood](configs/Parflood) folder).

## Pre-training
To train the FloodSformer model we initialize the parameters with the weights pretrained on the MovingMNIST dataset by *[Ye & Bilodeau, (2023)](https://doi.org/10.1016/j.imavis.2022.104612)* ([Github repository](https://github.com/XiYe20/VPTR)).

The following table provides the pre-trained checkpoints on MovingMNIST dataset. Use them to re-train the FloodSformer model.

| Model | Training phase | url |
| ---- | ---- | ---- |
| Autoencoder | 1. AE training | [download](https://drive.google.com/uc?export=download&id=1O783CqcoAJ4Tt75wmZNutyeLaGArhras) |
| VPTR-FAR | 2. VPTR training | [download](https://drive.google.com/uc?export=download&id=1F0W6jHgCZbrq_GfreHNEfy_i_AS0u4RJ) |


## Results
The following table provides the trained weights of the FloodSformer model for the different case studies. Use them to test the real-time forecasting application of the FloodSformer model.

| Case study | Input frames (I) | Autoencoder (AE) | Transformer (VPTR) |
| ---------- | :--------------: | :--------------: | :----------------: |
| 1. Dam-break in a parabolic channel | 8 | [download](https://drive.google.com/uc?export=download&id=1pXmRiZZu6j6f9b2piqIkKwgFp5AMsRk2) | [download](https://drive.google.com/uc?export=download&id=16K7sKMIYirHSXnO0OXjHs7YIaLwoV9jP) |
| 2. Dam-break in a rectangular tank | 8 | [download](https://drive.google.com/uc?export=download&id=1T6o5wLpQ-8ddAmYC99dd7ILS94wCaR0j) | [download](https://drive.google.com/uc?export=download&id=11FArxROwzM-lgs_l7eb2d0io8L3bNLEw) |
| 3a. Dam-break of the Parma River flood detention reservoir | 8 | [download](https://drive.google.com/uc?id=1vVHUywPkQQe-Mdp7bpGFA8dk4AyoilYq&export=download) | [download](https://drive.google.com/uc?id=1g227hmNiUWEa8ZQ8_qZZu22yptuPhVJG&export=download) |