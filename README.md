# Fashion-recommender-system

## Overview
This deep learning based project Fashion Recommender System suggests similar fashion items based on a given image. The system utilizes a pre-trained ResNet50 model for feature extraction from images and a K-Nearest Neighbors (KNN) algorithm to recommend the most similar items based on image embeddings.

## Features
- Image Upload: Users can upload an image of a fashion item.
- Feature Extraction: The uploaded image's features are extracted using the ResNet50 model.
- Recommendation: The system recommends the top 5 similar fashion items from a dataset based on Euclidean distance between the extracted features.
- User Interface: Built using Streamlit for a simple and interactive UI.
  
## Requirements
- Python 3.7+
- tensorflow for the model and feature extraction.
- streamlit for the user interface.
- scikit-learn for the nearest neighbor search.
- PIL for image processing.
- tqdm for progress bars during feature extraction.

## How It Works
1. Model Setup: The ResNet50 model (pre-trained on ImageNet) is used to extract features from images. The top layer is removed, and a GlobalMaxPooling2D layer is added to generate the final image embeddings.
2. Feature Extraction: For each fashion item in the dataset, features are extracted and stored in embeddings.pkl. The uploaded image's features are compared to these stored embeddings using the K-Nearest Neighbors (KNN) algorithm.
3. Recommendations: When a user uploads an image, the system computes the Euclidean distance between the extracted features and the stored embeddings. The closest items are returned as recommendations.

Dataset Link :- https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
