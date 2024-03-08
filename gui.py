import numpy as np
import cv2
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam,SGD
from keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# # Reading images from RAVDESS dataset
# data_dir = 'rav'
# img_size = (64,64)

# train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.1)
# val_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.1)

# train_generator = train_datagen.flow_from_directory(
#         data_dir,
#         target_size=img_size,
#         batch_size=64,
#         subset='training',
#         color_mode="grayscale",
#         class_mode='categorical')

# validation_generator = val_datagen.flow_from_directory(
#         data_dir,
#         target_size=img_size,
#         batch_size=64,
#         subset='validation',
#         color_mode="grayscale",
#         class_mode='categorical')

# emotion_dict = {0: "Angry", 1: "Calm", 2: "Fearful", 3: "Happy", 4: "Sad"}
# emoji_dist={0:"./emojis/angry.png",1:"./emojis/neutral.png",2:"./emojis/fearful.png",
#             3:"./emojis/happy.png",4:"./emojis/sad.png"}


# Reading images from FER-2013 dataset
train_dir = 'fer/train'
val_dir = 'fer/test'
img_size = (48,48)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png",
            3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}


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
    emotion_model.add(Dropout(0.25))

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


# get the model and load weights
model = get_model2(num_classes,img_size)
model.load_weights('weights_fer2.h5')


# gui
import tkinter as tk
from PIL import Image, ImageTk

cv2.ocl.setUseOpenCL(False)

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0,0]
cap1 = cv2.VideoCapture(0)
def show_vid():
    if not cap1.isOpened():
        print("cant open the camera")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(600,500))
    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=1)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, img_size), -1), 0)
        prediction = model.predict(cropped_img)
        show_text[1] = np.max(prediction)
        maxindex = int(np.argmax(prediction))
        show_text[0]=maxindex
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(25, show_vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def show_vid2():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(pic2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    lmain4.configure(text=f'Confidence = {str(int(show_text[1] * 100))} %.',font=('arial',45,'bold'))
    lmain2.configure(image=imgtk2)
    lmain2.after(1000, show_vid2)


root=tk.Tk() 

heading=tk.Label(root,text="Emojify",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 
heading.pack()

lmain = tk.Label(master=root,padx=50,bd=10)
lmain2 = tk.Label(master=root,bd=10)

lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
lmain4=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
lmain.pack(side=tk.LEFT)
lmain.place(x=50,y=250)
lmain3.pack()
lmain3.place(x=960,y=250)
lmain4.pack()
lmain4.place(x=50,y=150)
lmain2.pack(side=tk.RIGHT)
lmain2.place(x=900,y=350)

root.title("Emojify")
root.geometry("1400x900+100+10") 
root['bg']='black'
exitbutton = tk.Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = tk.BOTTOM)

show_vid()
show_vid2()

root.mainloop()

cap1.release()