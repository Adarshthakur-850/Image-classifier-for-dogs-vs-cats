import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

TRAIN_DIR = os.path.join("data", "cats_dogs_light", "train")
TEST_DIR = os.path.join("data", "cats_dogs_light", "test")

def create_generators(target_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, validation_generator, test_generator
