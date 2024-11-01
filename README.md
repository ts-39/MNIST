# MNIST Handwritten Digit Classification

## Overview
This project explores two different neural network architectures for classifying handwritten digits from the MNIST dataset. One implementation uses a one-dimensional neural network (`mnist.py`), and the other employs a Convolutional Neural Network (CNN) (`mnistcnn.py`). Through this project, I gained insights into how network architecture and batch size impact model performance on image classification tasks.

## Motivation
As a Data Science student with a keen interest in machine learning and image processing, this project allowed me to experiment with different neural network approaches to solve a well-known classification problem. Working with both a simple neural network and a CNN helped me better understand how architecture choices affect model accuracy, particularly when processing image data.

## Results

### 1. One-Dimensional Neural Network (`mnist.py`)
This code explores the performance of a one-dimensional neural network on image data. I found that **batch size had a significant effect on accuracy and loss**:

- **Batch size = 128**
  - Test loss: 2.3577
  - Test accuracy: 12.63%
  
- **Batch size = 2048**
  - Test loss: 0.2843
  - Test accuracy: 92.06%

These results highlighted the importance of batch size in training and its impact on the model's performance.

### 2. Convolutional Neural Network (`mnistcnn.py`)
This code implements a CNN model designed to improve accuracy on image classification. The CNN achieved notably better results:

- Test loss: 0.0219
- Test accuracy: 99.36%

This comparison taught me how much more effective CNNs are for image processing tasks compared to one-dimensional neural networks, which struggle with capturing spatial information in images.

## Dataset
The MNIST dataset is publicly available through `tensorflow.keras` and can be loaded directly in Python. To import the dataset, use the following code:

```python
from tensorflow.keras.datasets import mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## Installation
To run this project, you’ll need:
- Python 3.7 or above
- Essential libraries: TensorFlow, Keras, NumPy, and Matplotlib (other requirements are in `requirements.txt`)

## Usage
1. Clone the repository.
2. Install the required libraries with `pip install -r requirements.txt`.
3. Run `mnist.py` or `mnistcnn.py` to train and test the respective models on the MNIST dataset.

## Future Plans
Moving forward, I plan to further optimize the CNN model, experiment with additional architectures, and explore data augmentation techniques to enhance model performance.

## Acknowledgments
This project was a rewarding learning experience, and I’d like to thank the open-source community for the resources and support that guided me in developing and understanding these models.
