# 🏠 House Recognition (SuperAI Engineer Hackathon)

A high-performance Computer Vision pipeline built to classify house images. This project was developed during the SuperAI Engineer Hackathon and utilizes advanced training and inference strategies to achieve near-perfect accuracy.

## 🏆 Performance
* **Metric:** Accuracy Score
* **My Private Score:** `0.98727`
* **Competition Baseline:** `0.97636`
* **Result:** Successfully outperformed the baseline by utilizing modern CNN architectures and Test-Time Augmentation (TTA).

## 🛠️ Tech Stack
* **Deep Learning Framework:** `PyTorch`, `Torchvision`
* **Model Architecture:** `EfficientNetV2-S` (Pre-trained)
* **Data Processing:** `Pandas`, `Scikit-Learn` (Stratified Split), `PIL`
* **Optimization:** AdamW, CosineAnnealingLR

## ⚙️ Data Pipeline & Methodology

### 1. Robust Data Loading & Augmentation
* **Fault-Tolerant Loader:** Implemented a pre-scan masking technique to dynamically filter out missing or corrupted image paths from the CSV before training, preventing `KeyError` crashes.
* **Stratified Split:** Used `train_test_split` with stratification to ensure a balanced class distribution across training and validation sets.
* **Heavy Augmentation:** Applied `RandomCrop`, `RandomHorizontalFlip`, `RandomRotation(15)`, and `ColorJitter` to prevent overfitting.

### 2. Model Architecture & Training Strategy
* **Base Model:** Fine-tuned `EfficientNetV2-S`, replacing the final classifier head to output 2 classes.
* **Label Smoothing:** Applied `CrossEntropyLoss` with `label_smoothing=0.1` to penalize overconfident predictions and improve generalization.
* **Learning Rate Scheduling:** Utilized `AdamW` optimizer combined with a `CosineAnnealingLR` scheduler over 15 epochs for smooth convergence.

### 3. Advanced Inference: Test-Time Augmentation (TTA)
To squeeze out maximum performance and stabilize predictions on the hidden test set, we implemented TTA:
* For every image in the test set, the model generated predictions for both the **original image** and a **horizontally flipped version** (`torch.flip(inputs, dims=[3])`).
* The softmax probabilities from both variations were averaged (`avg_prob = (prob1 + prob2) / 2.0`) before determining the final class. This single technique significantly reduced error rates.

## 🚀 How to Run
1. Install dependencies: `pip install torch torchvision pandas scikit-learn pillow tqdm`
2. Prepare your dataset in the `/content/dataset` directory (Train images, Test images, and CSV files).
3. Run the Jupyter Notebook cell by cell to train the model and generate the `submission_TTA.csv` file.
