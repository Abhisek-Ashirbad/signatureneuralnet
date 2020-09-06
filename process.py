import os
import numpy as np
from os import listdir
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Functions

#base-dir
base_dir = 'C:/Users/Abhisek/Desktop/Programs/imagenet2/'

#training datasets
train_dir = os.path.join(base_dir, 'train')
train_real_dir1 = os.path.join(train_dir, 'dataset1/real')
train_dupe_dir1 = os.path.join(train_dir, 'dataset1/dupe')

train_real_dir2 = os.path.join(train_dir, 'dataset2/real')
train_dupe_dir2 = os.path.join(train_dir, 'dataset2/dupe')

train_real_dir3 = os.path.join(train_dir, 'dataset3/real')
train_dupe_dir3 = os.path.join(train_dir, 'dataset3/dupe')

train_real_dir4 = os.path.join(train_dir, 'dataset4/real')
train_dupe_dir4 = os.path.join(train_dir, 'dataset4/dupe')

#validation datasets
validation_dir = os.path.join(base_dir, 'validation')
validation_real_dir = os.path.join(validation_dir, 'real')
validation_dupe_dir = os.path.join(validation_dir, 'dupe')

train_datagen = ImageDataGenerator(rescale = 1.0/255.)
validation_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    batch_size = 10, 
                                                    class_mode = 'binary', 
                                                    target_size = (150, 150))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                batch_size = 10, 
                                                                class_mode = 'binary', 
                                                                target_size = (150, 150))

#Deep-learning model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#Model compilation
model.compile(optimizer = RMSprop(lr = 0.001), loss = 'binary_crossentropy', metrics = ['acc'])

#Training the above model
history = model.fit_generator(train_generator, 
                                validation_data = validation_generator, 
                                steps_per_epoch = 72, 
                                epochs = 15,
                                validation_steps = 10,
                                verbose = 1)