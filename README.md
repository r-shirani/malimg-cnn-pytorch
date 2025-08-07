
# CNN-Based Malware Image Classification with PyTorch

## Overview

This project focuses on building and training a **Convolutional Neural Network (CNN)** using **PyTorch** to classify grayscale malware images from the **Malimg dataset**. The dataset contains binary executable files transformed into 64x64 grayscale images, each representing a sample from a specific malware family.

The objective is to design a simple CNN architecture and train it end-to-end to distinguish between multiple malware classes.

---

## Dataset: Malimg

- The **Malimg dataset** consists of malware binary files converted to grayscale images.
- Each image is **64x64 pixels** in size and labeled with its malware family.
- Classes are **unbalanced**, requiring careful model evaluation.

---

## CNN Architecture

The CNN model designed in this project consists of the following layers:

1. **Convolution Layer 1**
   - Filters: 16
   - Kernel Size: 3x3
   - Padding: 1
   - Activation: ReLU

2. **Max Pooling Layer 1**
   - Kernel Size: 2x2
   - Stride: 2

3. **Convolution Layer 2**
   - Filters: 32
   - Kernel Size: 3x3
   - Padding: 1
   - Activation: ReLU

4. **Max Pooling Layer 2**
   - Kernel Size: 2x2
   - Stride: 2

5. **Fully Connected Layer**
   - Units: 128
   - Activation: ReLU

6. **Output Layer**
   - Units: Equal to the number of classes
   - Activation: Softmax (via `CrossEntropyLoss`)

---

## Hyperparameters

| Hyperparameter      | Value        | Justification |
|---------------------|--------------|---------------|
| Learning Rate       | 0.001        | Standard initial value for Adam optimizer |
| Optimizer           | Adam         | Adaptive learning with good convergence |
| Loss Function       | CrossEntropyLoss | Suitable for multi-class classification |
| Epochs              | 20           | Sufficient for observing convergence trends |
| Batch Size          | 64           | Balanced choice for speed and stability |
| Kernel Size         | 3x3          | Captures local patterns while maintaining resolution |
| Padding             | 1            | Preserves input size after convolution |
| Pooling Type        | MaxPooling   | Helps in capturing dominant features |

---

## Training & Evaluation

- Data is loaded using custom PyTorch datasets and loaders.
- Images are normalized and transformed into tensors.
- The model is trained for 20 epochs using the Adam optimizer.
- Evaluation is performed on a separate test set.

### Final Accuracy

- **Test Accuracy**: *Reported in the notebook output (please check final cell)*

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/malimg-cnn-classification.git
   cd malimg-cnn-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset (download from Kaggle using `kaggle.json` and extract to the proper folder structure).

4. Run the notebook:
   ```bash
   Jupyter Notebook HW7_1.ipynb
   ```

---

## File Structure

```
malimg-cnn-pytorch
│
├── HW7_1.ipynb           # Main notebook with CNN implementation and training
├── README.md             # Project documentation
├── requirements.txt      # Dependencies (optional)
└── malimg_dataset/       # Folder containing images (organized by class)
```

---

## Notes

- The model can be further improved using data augmentation and deeper architectures.
- Evaluation metrics such as precision, recall, and F1-score can be added for detailed performance insights.
- For a production-grade classifier, consider experimenting with CNN regularization techniques.

---

## License

This project is intended for educational purposes. The Malimg dataset is used under the terms outlined on Kaggle.
