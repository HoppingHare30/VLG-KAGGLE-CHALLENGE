# VLG-KAGGLE-CHALLENGE
This repository contains code for my submission for Pixel Play'25. The goal of this project is to classify animal species from images into 40 categories.

Prerequisites:

Python 3.8 or later

TensorFlow

Keras

NumPy

Pandas

Matplotlib

Training Configuration

Base Model:

Pre-trained Inception V3 (ImageNet weights, base layers frozen).

Custom Layers:

Global Average Pooling layer.

Fully connected dense layer with 1024 units.

Dropout layers to reduce overfitting.

Output layer with softmax activation.

Training Setup:

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

Epochs: 30

Batch size: 32

Callbacks:

ModelCheckpoint: Saved the best-performing model.

ReduceLROnPlateau: Adjusted learning rate upon validation plateau.

EarlyStopping: Halted training when validation performance stagnated.


