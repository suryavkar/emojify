import numpy as np
import cv2
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam,SGD
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


# Reading images from RAVDESS dataset
data_dir = 'rav'
img_size = (64,64)

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.1)
val_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=64,
        subset='training',
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=64,
        subset='validation',
        color_mode="grayscale",
        class_mode='categorical')


# # Reading images from FER-2013 dataset
# train_dir = 'fer/train'
# val_dir = 'fer/test'
# img_size = (48,48)

# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=img_size,
#         batch_size=64,
#         color_mode="grayscale",
#         class_mode='categorical')

# validation_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=img_size,
#         batch_size=64,
#         color_mode="grayscale",
#         class_mode='categorical')


num_classes = len(train_generator.class_indices);
print("Class labels",train_generator.class_indices)


# defining the model
def get_model1(num_class,img_size):
    input_size = img_size + (1,)
    emotion_model = Sequential()

    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_size))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.5))

    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(num_class, activation='softmax'))

    emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001, decay=1e-6),
                          metrics=['accuracy'])
    return emotion_model


# defining the model
def get_model2(num_class,img_size):
    input_size = img_size + (1,)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_size))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# get the model
emotion_model = get_model2(num_classes,img_size);
emotion_model.summary()


# train the model and store training info
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=11575 // 64,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=1284 // 64,
        shuffle=True)


# VISUALISING ACCURACY AND LOSS VS ITERATIONS
acc = [0.] + emotion_model_info.history['accuracy']
val_acc = [0.] + emotion_model_info.history['val_accuracy']
loss = emotion_model_info.history['loss']
val_loss = emotion_model_info.history['val_loss']
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='upper left')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Categorical Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# store trained weights
emotion_model.save_weights('weights_rav2.h5')