# Flower Species Classification using Deep Learning
## 📝 Project Overview
This project implements and compares various deep learning approaches for automatic flower species classification. The system classifies five common flower species (daisy, dandelion, rose, sunflower, and tulip) using Convolutional Neural Networks (CNNs) and transfer learning techniques.
## 🎯 Objectives

Develop robust automated flower species identification system
Compare baseline CNN vs. deeper CNN with regularization
Implement transfer learning using pre-trained ResNet50
Analyze computational efficiency and training performance
Address class imbalance in flower image datasets

## 🌺 Dataset

Total Images: 4,317 RGB flower images
Classes: 5 flower species
Distribution:

Dandelion: 1,051 images (24.37%)
Tulip: 983 images (22.80%)
Daisy: 763 images (17.69%)
Rose: 783 images (18.16%)
Sunflower: 732 images (16.98%)


Train/Validation Split: 80/20 stratified split

Training: ~3,421 images
Validation: ~856 images



## 🔧 Technologies Used

- **Python 3.x**: Programming language
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy & Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Evaluation metrics
- **PIL (Pillow)**: Image handling

### 🏗️ Model Architectures
#### 1. Baseline CNN Model

4 convolutional blocks with increasing filters (32, 64, 128, 256)
Each block: 2 Conv2D layers + ReLU + MaxPooling2D
GlobalAveragePooling2D for feature aggregation
Dense layers with ReLU and Softmax
Achieved: 73.8% validation accuracy

#### 2. Deeper CNN with Regularization

Enhanced baseline architecture
Added Dropout (rate=0.3) for regularization
Same convolutional structure with improved training stability
Achieved: 77.9% validation accuracy
Training Speed: 2.6x faster than baseline

#### 3. Transfer Learning with ResNet50

Phase 1: Feature extraction with frozen ResNet50 layers
Phase 2: Fine-tuning entire network with low learning rate
Custom classification head with 512-unit dense layer
ReduceLROnPlateau for adaptive learning rate
Achieved: 79.1% validation accuracy

## 📈 Results Summary
### Model Validation Metrics

| Model                          | Validation Accuracy | Validation Loss | Training Time/Epoch |
|--------------------------------|---------------------|-----------------|---------------------|
| Baseline CNN                   | 73.8%              | 0.70            | 2.39 seconds        |
| Deeper CNN + Regularization    | 77.9%              | 0.62            | 8.55 seconds        |
| ResNet50 Transfer Learning     | 79.1%              | 0.87            | ~5-20 seconds       |

### Class-Specific Performance (ResNet50)

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Daisy      | 0.82      | 0.79   | 0.80     |
| Dandelion  | 0.83      | 0.83   | 0.83     |
| Rose       | 0.77      | 0.71   | 0.74     |
| Sunflower  | 0.81      | 0.83   | 0.82     |
| Tulip      | 0.73      | 0.78   | 0.76     |
