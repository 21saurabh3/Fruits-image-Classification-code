# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 00:10:40 2022

@author: Saurabh
"""

## We are loading the model pretrained on imagenet. loading it with correct weights,set an input shape,choose to remove last layer of the model..
## Images have three dimensions height, width and a no of channels. There are three channel for three color.
## so there will be three class.


from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)


## Freezing the base model is done so that all the learning from imagenet dataset does not get destroyed in initial training


# Freeze base model
base_model.trainable = False


# Create inputs with correct shape
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(1, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)


model.summary()

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'] )

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
       samplewise_center= True,
       rotation_range = 10,
       zoom_range = 0.1,
       width_shift_range  = 0.1,
       height_shift_range = 0.1,
       horizontal_flip = True,
       vertical_flip  = False,
)
datagen_valid = ImageDataGenerator(samplewise_center= True)


# load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    "D:/fruits_project/archive/dataset/dataset/train/",
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
)
# load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
   "D:/fruits_project/archive/dataset/dataset/test/" ,
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
)

## Train the model

model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=20)


# Unfreeze the model for fine tuning it

# Unfreeze the base model 

base_model.trainable = True

# Compile the model with a low learning rate
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),
              loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=10)

model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)