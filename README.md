# CNN Based Image Classifier

This repository contains my implementation for **Assignment 2 - Part A** of the **DA6401 Deep Learning Systems** course. The goal of this assignment is twofold:

1. Train a Convolutional Neural Network (CNN) model **from scratch** on a subset of the **iNaturalist** dataset.
2. **Fine-tune a pre-trained model** using best practices like sweeps and experiment tracking with **Weights & Biases (wandb)**.

---

# Repository Structure
assignment_2_PartA/ ├── question_1/ │ ├── custom_cnn.py # Custom CNN model with configurable layers │ ├── train.py # Training code from scratch │ ├── requirements.txt # All necessary packages │ └── Readme.md # Sub-readme for Q1 │ ├── question_2/ │ ├── model.py # Flexible CNN with sweep capabilities │ ├── train.py # Training script with wandb integration │ ├── main.py # Runs training + validation loop │ ├── sweep_config.py # Wandb hyperparameter sweep configuration │ ├── evaluate.py # Final evaluation on test set │ └── .gitkeep │ ├── question_4/ │ ├── best_model.py # Best model architecture + weights │ ├── evaluation.py # Final evaluation code on test set │ └── .gitkeep


---

#  Instructions to Run

# Step 1: Clone the Repository

```bash
git clone https://github.com/Neel28iitm/assignment_2_PartA.git
cd assignment_2_PartA

# Step 2: Create Virtual Environment
python -m venv venv
source venv/bin/activate    

# Step 3: Install Dependencies
pip install -r question_1/requirements.txt

# Dataset
We used a subset of the iNaturalist dataset. The data was split into:
80% Training
20% Validation (class-balanced)
Test set was untouched until final evaluation
