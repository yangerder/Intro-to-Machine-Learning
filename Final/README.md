---

# Bagging-Based Emotion Classification

This repository contains code for training and inferring a Bagging-based ensemble of ResNet18 models for emotion classification.

## Setup Environment

1. Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

2. Create the environment using the provided environment file:
   ```
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```
   conda activate 111550149-ML
   ```
4. Install additional dependencies (if needed):
   ```
   conda install numpy pandas matplotlib scikit-learn pytorch torchvision -c pytorch
   conda install opencv
   ```
5. if not activate 111550149-ML:
   ```
   1.Enter ctrl+shift+P
   2.select interpreter
   3.choose 111550149-ML
   4.reopen the terminal
   ```


## Data Preparation

1. Place your dataset in a folder named `data` at the same directory level as the code files.
   - Inside `data`, ensure there are subdirectories for each emotion category (e.g., `happy`, `sad`, `angry`).
   - Train and test datasets should follow this structure:
     ```
     data/
       Images/
         train/
           happy/
           sad/
           ...
         test/
           ...
     ```

## Training

1. Run the training script to train the Bagging models:
   ```
   python Training.py
   ```

   This will:
   - Train 6 ResNet18 models.
   - Save the trained models' weights as `resnet18_bagging_1.pth`, ..., `resnet18_bagging_6.pth`.

## Inference

1. You can download the model from [here](https://drive.google.com/drive/folders/1OFLyQxVVXZaFfS0cFHwpwqRxbRSbobb7?usp=sharing)
2. Ensure the trained model weights are in the same directory as `inference.py`(there are six trained model weights!!!!!!!!!).
3. Run the inference script to predict the labels for the test dataset:
   ```
   python inference.py
   ```

   The predictions will be saved in a file named `submission_bagging.csv`.
