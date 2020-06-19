#!/usr/bin/python
#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Lambda, LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.layers.convolutional import Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import helper
import pickle
import math
#defien hyper_parameters
batch_size = 32
epochs = 5
drop_prob = 0.55
input_shape = (160, 320, 3)
# load the preprocess data
# dataset = pickle.load(open('./dataset5', 'rb'))
# x_input = np.array(dataset['images'])
# labels = np.array(dataset['labels'])

## prepare data ##
train_center_path = './generator_data/training_data/center_line' #the data set of keeping driving in center line
train_reco_path = './generator_data/training_data/recovery'      # the data set of keeping driving along the edge line 
train_anti_path = './generator_data/training_data/anti'          # the data set of keeping driving in anti-clockwise derection
train_open_path = './generator_data/training_data/open_aera'     # the data set of avoiding driving into open area
train_mid_reco_path = './generator_data/training_data/mid_reco' # the data set of driving close to the edge line and  recovery to center line
train_big_reco_path = './generator_data/training_data/big_reco'  # the data set of driving cross the edge line and recovery to center line
valid_path ='./generator_data/validation_data'



train_samples = helper.get_samples(train_center_path)
train_samples_filter = helper.filter_samples(train_samples, 0)

train_reco_samples = helper.get_samples(train_reco_path)
train_reco_filter = helper.filter_samples(train_samples_filter, 1)

train_open_samples = helper.get_samples(train_open_path)
train_open_filter = helper.filter_samples(train_open_samples, 1)

train_mid_reco_samples = helper.get_samples(train_mid_reco_path)
train_mid_filter = helper.filter_samples(train_mid_reco_samples, 1)

train_big_reco_samples = helper.get_samples(train_big_reco_path)
train_big_filter = helper.filter_samples(train_big_reco_samples, 1)

anticlock_samples = helper.get_samples(train_anti_path)
anticlock_samples_filter = helper.filter_samples(anticlock_samples, 0)

valid_samples_filter = helper.get_samples(valid_path)


train_add_anticlock = train_samples_filter + train_open_filter + train_mid_filter + 2*train_big_filter + anticlock_samples_filter#+ train_big_filter#+   +train_reco_filter + anticlock_samples_filter+ train_big_filter*2 
# print('test here --------------train_samples_fliter----------------------test here')
# print(len(train_samples_filter))
# print(len(train_add_anticlock))
# valid_samples = helper.get_sample(valid_path) 
train_generator = helper.generator(train_add_anticlock, batch_size= batch_size)
valid_generator = helper.generator(valid_samples_filter, batch_size= batch_size)
# print(train_generator)

# x_valid = x_input[2000:2800]
# y_valid = labels[2000:2800]

# ## apply the lenet-5 arch.##
# # create the Seuential model
# model = Sequential()
# model.add(Lambda(lambda x: x /255 -0.5, input_shape = (160, 320, 3)))
# model.add(Cropping2D(cropping=((75, 20), (0, 0))))
# model.add(Conv2D(6, kernel_size=(5, 5), input_shape=input_shape))
# # model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.05))
# # TO DO check valid or same
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(16, kernel_size=(5, 5), input_shape=input_shape))
# # model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.05))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten(input_shape=(160, 320, 3)))
# model.add(Dense(120, activation = 'relu'))
# model.add(Dropout(drop_prob))
# model.add(Dense(84, activation = 'relu'))
# model.add(Dropout(drop_prob))
# model.add(Dense(1))
##########   try nvidia cnn network
model = Sequential()
model.add(Lambda(lambda x: x /255 -0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((75, 20), (0, 0))))
model.add(Conv2D(24, kernel_size=(5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(36, kernel_size=(5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, kernel_size=(5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
# model.add(Dense(1164))
# model.add(Dropout(drop_prob))
model.add(Dense(100))
model.add(Dropout(drop_prob))
model.add(Dense(50))
model.add(Dropout(drop_prob))
model.add(Dense(10))
model.add(Dropout(drop_prob))
model.add(Dense(1))
############ try nivida net

model.summary()

# Compile the model
model.compile(optimizer='Adam', loss='mse')
# avoid overfitting
checkpoint = ModelCheckpoint(filepath = './', monitor = 'val_loss', save_best_only = True)
stopper = EarlyStopping(monitor = 'val_acc', min_delta= 0.0003, patience = 3)
ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
# datagen = ImageDataGenerator(
#         rotation_range=1,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         shear_range=0.1,
#         zoom_range=0.1,
#         horizontal_flip=False,
#         fill_mode='nearest', validation_split= 0.2)
# datagen.fit(x_train)
history_object = model.fit_generator(train_generator,
                steps_per_epoch=math.ceil(len(train_add_anticlock)/batch_size), 
                validation_data= valid_generator,
                validation_steps= math.ceil(len(valid_samples_filter)/batch_size),
                epochs = 7, verbose = 1)
# model.fit(x_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 7,batch_size = 128)

# save model
model.save('model.h5')

#visualize the loss
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
