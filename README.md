# Face Recognition Project

This project demonstrates how to use a face recognition model built with TensorFlow, Keras, and OpenCV. It provides the ability to train a model on a custom dataset, save it, and later use it for real-time face recognition using a webcam.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Dataset Structure](#dataset-structure)
3. [Installation](#installation)
4. [Training the Model](#training-the-model)
5. [Testing the Model](#testing-the-model)
6. [Using the Model](#using-the-model)
7. [Model Architecture](#model-architecture)
8. [Save and Load the Model](#save-and-load-the-model)
9. [Contributing](#contributing)
10. [License](#license)

## Prerequisites

Before you start, make sure you have the following installed:

- Python 3.7 or later
- TensorFlow 2.x
- OpenCV
- Other dependencies listed in `requirements.txt`

## Dataset Structure

The dataset should be structured as follows: 

dataset/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... `use atlest 50 images of each person.`
└── Person2/
    ├── image1.jpg
    ├── image2.jpg
    └── ...`use atlest 50 images of each person .`

## Installation

Follow these steps to set up the environment and run the project:

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/Face-Recognition.git
cd Face-Recognition
```
### 2.Install requirements
```bash

pip install -r requirements.txt

pip install tensorflow opencv-python scikit-learn facenet-pytorch numpy

```

### 3. Prepare data set 

``` note - use atlest 50 images with different angels for best results ```

### 4.Train the model 

``` bash 

python train_model.py


```

### 5.Test the model 

```bash 

python test.py

```

### 6.Use model in your Script 

For refrence you can use file [Use_model.py]



## **Model Architecture**

The model is a simple Convolutional Neural Network (CNN) designed for face recognition. It uses the following layers:

1. **Input Layer**: Takes input images of size 160x160x3 (RGB images).
2. **Convolutional Layers**: 
   - 3 convolutional layers with 32, 64, and 128 filters respectively. Each layer uses ReLU activation functions and `same` padding.
   - These layers are followed by max-pooling to down-sample the feature maps.
3. **Flatten Layer**: The output of the final convolutional layer is flattened into a 1D vector.
4. **Dense Layers**:
   - A fully connected dense layer with 512 units and ReLU activation.
   - A dropout layer with a 50% dropout rate to prevent overfitting.
5. **Output Layer**: A softmax activation function is used in the output layer to classify the input image into one of the labels in the dataset.

The model is compiled with the **Adam optimizer** and **sparse categorical cross-entropy loss** for multi-class classification.

Here is the code for the model architecture:

```python
from tensorflow.keras import layers, models

def create_model(input_shape=(160, 160, 3)):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

### **Contributing**

We welcome contributions to improve this project! Whether it's bug fixes, new features, improvements to documentation, or other enhancements, your contributions are appreciated. Here’s how you can contribute:

#### How to Contribute

1. **Fork the Repository**  
   Click on the "Fork" button on the top-right of the repository page to create your own copy of the repository.

2. **Clone the Forked Repository**  
   Clone your forked repository to your local machine using the following command:

   ```bash
   https://github.com/Arya182-ui/Face_Recognition_Model.git
  


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Author

[Ayush Gangwar](https://github.com/Arya182-ui)
