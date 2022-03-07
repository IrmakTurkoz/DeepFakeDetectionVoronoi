import numpy as np
import os
import cv2
import sys
from shutil import copyfile
# Importing all necessary libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
import keras,os
import matplotlib.pyplot as plt
  
img_width, img_height = 224, 224

import DeepFakeDetection as model

def main():

#      fileDirectory = "New_new_Sample_set/val/manipulated_sequences/features/"
# # #     fileDirectory2 = "C:\\Irmak Dosyalar\\HW\\5.2\\CS 564\\Project\\Sample_set\\manipulated_sequences\\DeepFakeDetection\\c23\\videos"

#      destination = "New_new_Sample_set/val/manipulated_sequences/padded_features/"
# #     videos = [f for f in os.listdir(fileDirectory) if ("mp4") in f]
# #     print(videos)
# #     model.extract_features(fileDirectory,destination,videos,True,show_results =False,frame_rate = 50)
#      model.pad_images(fileDirectory,destination)

#     originals = [( f ,0) for f in os.listdir('original_features')if ("jpg") in f]
#     manipulateds = [( f,1) for f in os.listdir('manipulated_features')if ("jpg") in f]
#     np.random.shuffle(originals)
#     np.random.shuffle(manipulateds)
#     train_data = []
#     val_data = []
#     test_data = []
#     train_data.extend(originals[0: int(len(originals)*0.6)])
#     train_data.extend(manipulateds[0: int(len(manipulateds)*0.6)])
#     np.random.shuffle(train_data)
#     val_data.extend(originals[int(len(originals)*0.6):int(len(originals)*0.8)])
#     val_data.extend(manipulateds[int(len(manipulateds)*0.6):int(len(manipulateds)*0.8)])
#     np.random.shuffle(val_data)
#     test_data.extend(originals[int(len(originals)*0.8):int(len(originals))])
#     test_data.extend(manipulateds[int(len(manipulateds)*0.8):int(len(manipulateds))])
#     np.random.shuffle(test_data)
    
    
    # for (filename,label) in train_data:
    #     if(label == 0):
    #         copyfile("original_features/" + filename,"data/train/original/"+filename)
    #     else:
    #         copyfile("manipulated_features/" + filename,"data/train/manipulated/"+filename)

    # for (filename,label) in val_data:
    #     if(label == 0):
    #         copyfile("original_features/" + filename,"data/val/original/"+filename)
    #     else:
    #         copyfile("manipulated_features/" + filename,"data/val/manipulated/"+filename)

    # for (filename,label) in test_data:
    #     if(label == 0):
    #         copyfile("original_features/" + filename,"data/test/original/"+filename)
    #     else:
    #         copyfile("manipulated_features/" + filename,"data/test/manipulated/"+filename)
#     print("Train_data size: ",len(train_data))
#     print("Val data size: ",len(val_data))
#     print("Test data size: ",len(test_data))

    nb_train_samples = 4103 
    nb_validation_samples = 1732
    epochs = 24
    batch_size = 16
    img_width, img_height = 80, 100

    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3) 


    model = Sequential() 
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten()) 
    model.add(Dense(32)) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(units=2)) 
    model.add(Activation('softmax')) 
    # model = Sequential() 
    # model.add(Conv2D(16, (2, 2), input_shape=input_shape)) 
    # model.add(Activation('relu')) 
    # model.add(MaxPool2D(pool_size=(2, 2))) 
    
    # model.add(Conv2D(16, (2, 2))) 
    # model.add(Activation('relu')) 
    # model.add(MaxPool2D(pool_size=(2, 2))) 
    
    # model.add(Conv2D(32, (2, 2))) 
    # model.add(Activation('relu')) 
    # model.add(MaxPool2D(pool_size=(2, 2))) 
    
    # model.add(Flatten()) 
    # model.add(Dense(32)) 
    # model.add(Activation('relu')) 
    # model.add(Dropout(0.5)) 
    # model.add(Dense(units=2)) 
    # model.add(Activation('sigmoid')) 
    # from keras.optimizers import Adam
    # opt = Adam(lr=0.0045)
    model.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy']) 
    model.summary()
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator()

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator()

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'data/train',  # this is the target directory
            target_size=(80, 100),  # all images will be resized to 150x150
            batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'data/val',
            target_size=(80, 100),
            batch_size=batch_size)
    # this is a similar generator, for validation data
    test_generator = test_datagen.flow_from_directory(
            'data/test',
            target_size=(80, 100),
            batch_size=batch_size)

    history = model.fit_generator( 
            train_generator, 
            steps_per_epoch=nb_train_samples // batch_size, 
            epochs=epochs, 
            validation_data=validation_generator, 
            validation_steps=nb_validation_samples // batch_size) 
    predictions = model.predict(test_generator)
    model.save_weights('second_try.h5') 
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
main()


'''    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    '''
