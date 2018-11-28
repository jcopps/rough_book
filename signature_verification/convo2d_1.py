
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,  Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

forged_dir = 'axis_data_set/forged/'
genuine_dir = 'axis_data_set/genuine/'



%matplotlib inline

!pip install -q -U tensorflow>=1.8.0
import tensorflow as tf

from keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np


    

def filter_by_str(items, suffix):
    return [x for x in items if suffix in x]

def load_and_show_image(img_pat):
    img = image.load_img(img_pat, target_size=input_shape)
    #print(img_pat)
    x = image.img_to_array(img)
    #print(x)
    #print(x.shape)
    plt.imshow(x/255.)
    
    
def load_image(img_pat):
    img = image.load_img(img_pat, target_size=input_shape)
    x = image.img_to_array(img)
    image_name = img_pat[img_pat.rfind('/')+1:].split('.')[0].split('-')[1]
    img_label = image_name[:3]+image_name[5:]
    plt.imshow(x/255.)
    if img_label == '002002':
        return x, 1
    else:
        return x, 2
    #return x, img_label
                                                  
def load_images(path, data, data_label, label_int, filter_str=''):
    items = os.listdir(path)
    #print (items)
    #print len(items)
    filtered_images = filter_by_str(items, filter_str)
    print filtered_images
    for item in filtered_images:

        img, img_label = load_image(path+item)
        #print img_label

        data.append(label_int)
        data_label.append(img_label)

def train_and_test(filter_str):
  
    img_height, img_width = 224, 224
    input_shape = (img_height, img_width, 3)
    num_classes = 3
    epochs = 10
    batch_size = 1


    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height) 
    else:
        input_shape = (img_width, img_height, 3)

    training_data = []
    training_data_labels = []
    load_images(genuine_dir, training_data, training_data_labels, 1, filter_str)
    load_images(forged_dir, training_data, training_data_labels, 2, filter_str)
    #load_images(forged_dir, training_data, training_data_labels)

    mlb = MultiLabelBinarizer()

    training_data = np.array(training_data)
    training_data = training_data.astype('float32') / 255


    training_data_labels = np.array(training_data_labels)
    print "Labels: ", training_data_labels
    training_data_labels = to_categorical(training_data_labels)
    print "ybinary: ", training_data_labels
    print len(training_data), len(training_data_labels)
    test_data = training_data
    test_data_labels = training_data_labels
    #training_data = training_data[:10]
    #training_data_labels = training_data_labels[:10]
    #training_data_labels = mlb.fit_transform(training_data_labels)
    #print "fitLabels: ", training_data_labels



    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    #model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


    #my_callbacks = [EarlyStopping(monitor="acc", patience=5, mode=max)]

    hist = model.fit(training_data, training_data_labels, batch_size=batch_size, epochs=epochs, verbose=1)#, callbacks=my_callbacks)
    score = model.evaluate(test_data, test_data_labels)
    print("Testing Loss:", score[0])
    print("Testing Accuracy:", score[1])
    
    
    
train_and_test('002.')

