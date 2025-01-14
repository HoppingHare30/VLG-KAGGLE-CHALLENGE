# Dataset : kaggle competitions download -c vlg-recruitment-24-challenge

import os  # provides functions to interact with the operating system
pathtodataset = "/Users/shagunbhatia30/Downloads/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/new"  # path to the dataset folder
trainpath = pathtodataset + "/train"  # path containing training images
valpath = pathtodataset + "/validate"  # path containing validation images
noofclasses = len(os.listdir(trainpath))  # number of sub-folders representing categories
print(noofclasses)  # print the number of categories

import matplotlib.pyplot as plt  # used for displaying images and generating plots
import matplotlib.image as mpimg  # allows reading images into arrays
import random  # for selecting random elements

def viewrandom(targetdir , targetclass):  # function to view random image for a given class
    targetpath = targetdir + "/" + targetclass  # derive the path for the target class
    img = random.choice(os.listdir(targetpath))  # randomly choose an image
    img_array = mpimg.imread(targetpath + "/" + img)  # read the chosen image into a NumPy array
    plt.imshow(img_array)  # display the image
    plt.title(targetclass)  # set the title as the class name
    plt.axis("off")  # hide the axis for clear display
    plt.show()  # show the image

    print(f"image shape : {img_array.shape}")  # print the shape of the image
    return img_array  # return the image array for further processing
#viewrandom(trainpath , "cow")


import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # provides utilities for image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)  # normalize pixel values for training data (0 to 1 range)
val_datagen = ImageDataGenerator(rescale=1./255)  # normalize pixel values for validation data

traindata = train_datagen.flow_from_directory(trainpath, target_size=(224,224), batch_size=32, class_mode='categorical')
valdata = val_datagen.flow_from_directory(valpath, target_size=(224,224), batch_size=32, class_mode='categorical')

basemodel = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)  # load pre-trained model
basemodel.trainable = False  # freeze base model weights to use it for feature extraction
inputs = tf.keras.Input(shape=(224,224,3), name='inputlayer')  # input layer placeholder for images
x = basemodel(inputs)  # pass inputs through the base model
print(x.shape)
x = tf.keras.layers.GlobalAveragePooling2D(name = "globalavglayer")(x)
print(x.shape)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
outputs = tf.keras.layers.Dense(noofclasses, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # configure model for training
print(model.summary())  # display the model architecture

EPOCHS=30
best_model_path = "best_model123.keras"

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

callbacks = [
    ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1  # Enables logging of model save
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=4,
        verbose=1,
        factor=0.1,
        min_lr=1e-6
    ), EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1)
]

print(f"Best model will be saved to: {best_model_path}")
history = model.fit(  # train the model using the datasets and callbacks
    traindata,  # training dataset
    validation_data=valdata,  # validation dataset
    epochs=EPOCHS,  # number of epochs to train
    callbacks=callbacks,  # pass the callbacks for checkpoints and learning rate adjustment
    steps_per_epoch=len(traindata),  # total steps in each epoch
    validation_steps=int(0.25 * len(valdata))  # evaluate on 25% of validation steps
)

# Debug and force save the model after training
print("Training completed.")
print(f"Forcing a model save to: {best_model_path}")
model.save(best_model_path)
print(f"Model saved successfully to: {best_model_path}")
print(model.evaluate())
def plot_graphs(history):  # function to visualize training metrics
    loss = history.history['loss']  # extract training loss
    val_loss = history.history['val_loss']  # extract validation loss
    acc = history.history['accuracy']  # extract training accuracy
    val_acc = history.history['val_accuracy']  # extract validation accuracy
    epochs = range(len(loss))  # number of completed epochs
    plt.figure(figsize=(10, 10))  # define figure size to display plots
    plt.subplot(1, 2, 1)  # first subplot for loss
    plt.plot(epochs, loss, 'r', label='Training loss')  # plot training loss
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')  # plot validation loss
    plt.title('Training and Validation Loss')  # set plot title
    plt.xlabel('Epochs')  # label x-axis
    plt.ylabel('Loss')  # label y-axis
    plt.legend()  # add legend
    plt.subplot(1, 2, 2)  # second subplot for accuracy
    plt.plot(epochs, acc, 'r', label='Training acc')  # plot training accuracy
    plt.plot(epochs, val_acc, 'bo', label='Validation acc')  # plot validation accuracy
    plt.title('Training and Validation Accuracy')  # set plot title
    plt.xlabel('Epochs')  # label x-axis
    plt.ylabel('Accuracy')  # label y-axis
    plt.legend()  # add legend

    plt.show()  # display the plots
    plt.savefig('plot.png')  # save the plots to file

    plot_graphs(history)
