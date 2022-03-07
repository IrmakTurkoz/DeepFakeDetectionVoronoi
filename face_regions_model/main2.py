

import numpy as np
import os
import cv2
import sys
# from shutil import copyfile
# # Importing all necessary libraries 
# from keras.preprocessing.image import ImageDataGenerator 
# from keras.models import Sequential 
# from keras.layers import Conv2D, MaxPooling2D 
# from keras.layers import Activation, Dropout, Flatten, Dense 
# from keras import backend as K 
import matplotlib.pyplot as plt

# Define Model here

from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import to_categorical
from keras.applications.vgg16  import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.optimizers import SGD
# snippet of using the LearningRateScheduler callback
from keras.callbacks import LearningRateScheduler
import keras_video
from keras import backend as K 


img_width, img_height = 224, 224

import DeepFakeDetection as model

def main():
        # str1 = "/train/manipulated"
        # fileDirectory = "new_data" + str1
        # destination = "faces_new_data"+ str1
        
        # videos = [f for f in os.listdir(fileDirectory) if ("mp4") in f]
        # print(videos)
        # model.extract_features(fileDirectory,destination,videos,True,show_results =False,frame_rate = 50)
        # # model.pad_images("original_features/")

        # originals = [( f ,0) for f in os.listdir('original_features')if ("jpg") in f]
        # manipulateds = [( f,1) for f in os.listdir('manipulated_features')if ("jpg") in f]
        # np.random.shuffle(originals)
        # np.random.shuffle(manipulateds)
        # train_data = []
        # val_data = []
        # test_data = []
        # train_data.extend(originals[0: int(len(originals)*0.6)])
        # train_data.extend(manipulateds[0: int(len(manipulateds)*0.6)])
        # np.random.shuffle(train_data)
        # val_data.extend(originals[int(len(originals)*0.6):int(len(originals)*0.8)])
        # val_data.extend(manipulateds[int(len(manipulateds)*0.6):int(len(manipulateds)*0.8)])
        # np.random.shuffle(val_data)
        # test_data.extend(originals[int(len(originals)*0.8):int(len(originals))])
        # test_data.extend(manipulateds[int(len(manipulateds)*0.8):int(len(manipulateds))])
        # np.random.shuffle(test_data)
        
        
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
        # print("Train_data size: ",len(train_data))
        # print("Val data size: ",len(val_data))
        # print("Test data size: ",len(test_data))

        # nb_train_samples = len(train_data) 
        # nb_validation_samples = len(val_data)
        epochs = 24
        batch_size = 128
        img_width, img_height = 224, 224

        if K.image_data_format() == 'channels_first': 
                input_shape = (3, img_width, img_height) 
        else: 
                input_shape = (img_width, img_height, 3) 

        def lr_scheduler(epoch, lr):
            decay_rate = 0.1
            decay_step = 90
            if epoch % decay_step == 0 and epoch:
                return lr * decay_rate
            return lr
   
        
        learning_rate = 0.1
        decay_rate = 5e-2
        momentum = 0.9
        opt = Adam(lr=learning_rate, decay=decay_rate)
        #opt = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
        vgg16_model = VGG16(
            include_top=True,
            input_tensor=None,
            input_shape= input_shape,
            weights= None,
            pooling='max',
            classes=2)

        vgg16_model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
        vgg16_model.summary()



        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator()

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator()



        train_generator = train_datagen.flow_from_directory(
            'faces_new_data/train',  # this is the target directory
            target_size=(224, 224),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode="categorical")  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
            'faces_new_data/val',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical")
        # this is a similar generator, for validation data
        test_generator = test_datagen.flow_from_directory(
            'faces_new_data/test',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical")
        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
       
        # this is a similar generator, for validation data
       

        history = vgg16_model.fit_generator( 
                train_generator, 
                steps_per_epoch=20, 
                epochs=epochs, 
                validation_data=validation_generator, 
                validation_steps=20,
                verbose=1) 
        predictions = model.predict(test_generator)
        vgg16_model.save_weights('second_try.h5') 
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

