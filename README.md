# Content-Based Image Retrieval (CBIR) for Face Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-99.63%25-brightgreen)

A deep learning-based CBIR system for face recognition using **CNNs**, **VGG16**, and **FaceNet**, optimized for celebrity face retrieval and classification.

**Full, Detailed report and walkthrough is available in Report.pdf**


## 📁 Dataset
**Source:** [Celebrity Face Dataset (Kaggle)](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset)  
**Structure:**  
- 31 celebrity folders (e.g., `Henry Cavill`, `Natalie Portman`)  
- 50–150 images per class (varied poses/lighting)  
- Cropped faces using MTCNN for focus  

## 🛠️ Methodology
### 1. **Custom CNN**
- **Architecture:**  
  `Conv2D → MaxPooling → BatchNorm → Dropout → Dense`  
- **Training:**  
  - 3 classes (Henry Cavill, Natalie Portman, Robert Downey Jr.)  
  - 30 epochs, Adam optimizer  
  - **Accuracy:** 81.48% (test set)  

### 2. **VGG16 Transfer Learning**
- **Preprocessing:** Data augmentation (rotation, flip, brightness)  
- **Fine-tuning:**  
  - Frozen base layers + custom dense layers  
  - **Accuracy:** 92.59% (test set)  
- **Limitation:** Overconfidence in misclassifications  

### 3. **FaceNet + SVM**
- **Embeddings:** 128D face features extracted via FaceNet  
- **Classifier:** SVM with linear kernel  
- **Accuracy:** **99.63%** (test set)  
- **Key Advantage:** Robust to pose/lighting variations  

## 📊 Key Results
| Model          | Test Accuracy | F1-Score | Specialization          |
|----------------|---------------|----------|-------------------------|
| Custom CNN     | 81.48%        | 0.81     | Basic face recognition  |
| VGG16          | 92.59%        | 0.93     | Transfer learning       |
| **FaceNet+SVM**| **99.63%**    | **0.99** | State-of-the-art performance |


## 💻 How to Run
1. Install dependencies:
```bash
pip install tensorflow keras-facenet scikit-learn mtcnn opencv-python matplotlib
