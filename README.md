# Skin_Cancer_Detection
# Skin Cancer Detection using CNN
This repository contains code for building a Convolutional Neural Network (CNN) model to detect skin cancer from images. The model is trained using the HAM10000 dataset, which includes dermatological images categorized by the diagnosis of skin diseases.

**Table of Contents**

**Overview**

**Dependencies**

**Data**

**Code Structure**

**Steps**

**Model Training**

**Results**

**Overview**

Skin cancer is one of the most common types of cancer worldwide. Early detection plays a critical role in improving survival rates. This project aims to build an image classification model that can help detect skin cancer from images using a Convolutional Neural Network (CNN).

The dataset used for this project is HAM10000, which is a publicly available collection of dermatology images with various skin disease labels. The model's goal is to classify images into various types of skin conditions such as melanoma, basal cell carcinoma, etc.

**Dataset :-** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

**Dependencies**

To run this code, you will need the following Python libraries:

tensorflow: For building and training the CNN model.
numpy: For numerical operations.
pandas: For handling data and metadata.
opencv-python: For image loading and preprocessing.
matplotlib and seaborn: For data visualization.
sklearn: For data splitting and label encoding.

**Data**

The dataset used in this project is HAM10000, a collection of images from dermatology clinics labeled with the corresponding skin condition diagnosis.

**Metadata:** 

The dataset includes a CSV file HAM10000_metadata.csv that contains information such as image IDs and diagnosis labels.

**Images:** 

The image files are stored in a directory, and each file corresponds to an image ID in the metadata.
Ensure you have the images and metadata in their correct paths before running the code.

**Code Structure**

The code is structured as follows:

**Data Loading and Preprocessing:** Code to load and preprocess the images, including resizing, normalization, and encoding the labels.
Exploratory Data Analysis (EDA): Basic visualizations to explore the distribution of the data and random samples of images with their corresponding labels.

**Model Building:** A CNN model built using TensorFlow/Keras for skin cancer classification.
Training: The CNN model is trained on the dataset.

**Evaluation and Visualization:** After training, the model is evaluated on the test set, and visualizations of training history (accuracy and loss) are plotted.

**Steps**

**Load Data:** Load the metadata CSV and image files.

**Preprocess Images:** Resize the images, normalize pixel values, and encode labels.

**EDA:** Visualize the distribution of skin conditions and display a few random images.

**Split Data:** Split the data into training and test sets.

**Model Building:** Build a CNN model using Keras.

**Train the Model**: Train the model using the training data.

**Evaluate:** Evaluate the model's performance on the test set.

**Visualization:** Plot training and validation accuracy/loss curves.

**Model Training**

The model is a CNN with the following architecture:

**Input Layer:** The input image shape is (224, 224, 3).
Conv2D + MaxPooling2D Layers: Multiple convolutional and pooling layers to extract features from the images.
**Flatten Layer:** Flatten the feature maps to a 1D vector.
**Fully Connected Dense Layers:** A dense layer followed by the output layer with a softmax activation for multi-class classification.
The model is trained using the Adam optimizer and categorical cross-entropy loss.

**Training Parameters:**
Epochs: 10
Batch Size: 32
Validation Split: 20% for validation

**Results**
After training the model, the test accuracy is printed, and the training/validation accuracy and loss are plotted over epochs.

You can expect the model's performance to be dependent on various factors like training time, dataset size, and model tuning.
