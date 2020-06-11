############################################################################
# File Name: rcnn.py                                               #
#                                                                          #
# Developer: Rahul Mehta                                                   #
#                                                                          #
# Designer: Debi Prasad Sahoo, Anshul Prakash Deshkar, Rahul Mehta         #
#                                                                          #
# (c)2016-2020 Copyright Protected,NetworkFinancials Inc.,San Jose(CA),USA #
#                                                                          #
############################################################################

#[RCNN TRAINING/MODEL OPEN-SOURCED FROM PRIYA DWIVEDI'S PROJECT IN ACKNOWLEDGEMENTS]
#[ALSO, https://medium.com/@1297rohit/transfer-learning-from-scratch-using-keras-339834b153b9. HIS VGG MODEL WAS TRIED & TESTED AS WELL, BUT THE GENERALIZATION
# ERROR SURPASSED THE GEN ERROR FOR THE ONE USED BELOW]

import numpy
import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

#[PRE-PROCESSING DIRECTIVES TO REMOVE SOME ERRORS THAT POPPED UP]

#[LOADING FILES]
files_train = 0
files_validation = 0

cwd = os.getcwd()
folder = 'data/train'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_train += len(files)

folder = 'data/test'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_validation += len(files)

#[SETTING KEY PARAMETERS]
img_width, img_height = 48, 48
train_data_dir = "data/train"
validation_data_dir = "data/test"
nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 15
num_classes = 2

#[SETTING KEY PARAMETERS FOR MODEL]
model = applications.VGG16(weights = "imagenet", include_top = False, input_shape = (img_width, img_height, 3))
for layer in model.layers[:10]:
    layer.trainable = False
x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"]) 

#[TRAINING GENERATORS]
train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, fill_mode = "nearest", zoom_range = 0.1, width_shift_range = 0.1, 
								   height_shift_range=0.1, rotation_range=5)

test_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, fill_mode = "nearest", zoom_range = 0.1, width_shift_range = 0.1, 
								   height_shift_range=0.1, rotation_range=5)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size = (img_height, img_width), batch_size = batch_size, class_mode = "categorical")
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size = (img_height, img_width), batch_size = batch_size, class_mode = "categorical")

#[CHECKPOINTS AND EARLY STOPPING]
checkpoint = ModelCheckpoint("car1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

history_object = model_final.fit_generator(train_generator, samples_per_epoch = nb_train_samples, epochs = epochs, validation_data = validation_generator,
										nb_val_samples = nb_validation_samples, callbacks = [checkpoint, early])

#[TO VISUALIZE MODEL ACCURACY/LOSS UNCOMMENT THE FOLLOWING]
plt.plot(history_object.history['acc'])
plt.plot(history_object.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
