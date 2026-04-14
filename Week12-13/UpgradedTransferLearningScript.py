import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, MobileNet, DenseNet121, imagenet_utils
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input

# Ensure TensorFlow is using the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid memory allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found. Using CPU.")

# Optionally, log device placement for debugging
# tf.debugging.set_log_device_placement(True)  # Comment this out to suppress device placement logs

# ! unzip dataset.zip

# Updated prepare_image function to use MobileNet-specific preprocess_input
def prepare_image(file):
    img_path = ''
    img = Image.open(img_path + file)
    size = (224, 224)
    img1 = ImageOps.fit(img, size)
    img_array = np.asarray(img1)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return mobilenet_preprocess_input(img_array_expanded_dims)

# Explicitly place operations on the GPU (optional)
with tf.device('/GPU:0'):
    mobile = MobileNet()

    preprocessed_image = prepare_image('dataset/with_mask/0-with-mask.jpg')
    predictions = mobile.predict(preprocessed_image)
    # print(results)  # Comment this out to suppress prediction output

tf.keras.utils.plot_model(mobile, show_shapes=True)

# Updated InceptionV3 model creation
def inceptionModel(height, width, print_summary=False):
    model = InceptionV3(weights='imagenet', include_top=False, input_shape=(height, width, 3))
    model.trainable = False
    output = GlobalMaxPooling2D()(model.output)
    model = Model(inputs=model.input, outputs=output)
    if print_summary:
        model.summary()
    return model

def getFeatureVector(model, image):
    featureVector = model.predict(image)
    return featureVector.flatten()

file_path = 'dataset/with_mask/0-with-mask.jpg'
width, height = 224, 224
image = cv2.resize(cv2.imread(file_path), (width, height))
image = np.expand_dims(image, axis=0)

model = inceptionModel(width, height)
encoding = getFeatureVector(model, image)

# Updated data augmentation and generator creation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/',
    batch_size=20,
    class_mode='binary',
    color_mode='rgb',
    target_size=(224, 224)
)

validation_generator = test_datagen.flow_from_directory(
    'dataset/',
    batch_size=20,
    class_mode='binary',
    color_mode='rgb',
    target_size=(224, 224)
)

# Updated MobileNet-based classifier
def mobileNetModel_classifier(height, width, print_summary=False):
    pre_trained_model = InceptionV3(input_shape=(height, width, 3), include_top=False, weights='imagenet')
    x = Flatten()(pre_trained_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(pre_trained_model.input, x)
    if print_summary:
        model.summary()
    return model

# Explicitly place operations on the GPU (optional)
with tf.device('/GPU:0'):
    modelClassifier = mobileNetModel_classifier(height, width)

    for layer in modelClassifier.layers[:310]:
        layer.trainable = False
    for layer in modelClassifier.layers[310:]:
        layer.trainable = True

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            val_acc = logs.get('val_accuracy')  # Updated to use 'val_accuracy' (correct metric name in TensorFlow)
            if val_acc is not None and val_acc > 0.99:
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True

    modelClassifier.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = myCallback()

    history = modelClassifier.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=5,
        epochs=10,
        validation_steps=10,
        verbose=1,  # Change to 1 for less detailed logs or 0 for no logs
        callbacks=[callbacks]
    )

    modelClassifier.save("myMaskClassifer_inceptionv3_transferLearning.h5")

# Updated inference function
def load_image(img_path, show=False):
    img = load_img(img_path, target_size=(224, 224))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    return img_tensor

img_path = 'dataset/with_mask/0-with-mask.jpg'
new_image = load_image(img_path)

pred = modelClassifier.predict(new_image)
pred
