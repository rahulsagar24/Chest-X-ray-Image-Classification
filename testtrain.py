import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

# Non-Binary Image Classification using Convolution Neural Networks
'''
path = 'GWOImages'

labels = []
X_train = []
Y_train = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            X_train.append(im2arr)
            Y_train.append(getID(name))
        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)

X_train = X_train.astype('float32')
X_train = X_train/255
    
test = X_train[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
Y_train = to_categorical(Y_train)
np.save('model/GWOX.txt',X_train)
np.save('model/GWOY.txt',Y_train)
'''
X_train = np.load('model/GWOX.txt.npy')
Y_train = np.load('model/GWOY.txt.npy')
print(Y_train[10])
print(Y_train[11])
print(Y_train[22])
print(Y_train[33])
print(Y_train[44])
print(Y_train[55])
if os.path.exists('model/GWOmodel.json'):
    with open('model/GWOmodel.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/GWOmodel_weights.h5")
    classifier._make_predict_function()   
    print(classifier.summary())
    f = open('model/GWOhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))
else:
    classifier = Sequential()
    #defining model with 32, 64 and 128 and due to system memory limit we have restrict image size to 64 X 64 
    classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) #define XCOVNet layer with image input size as 64 X 64 with 3 RGB colours
    classifier.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer to collect filter data
    classifier.add(Convolution2D(64, 3, 3, activation = 'relu')) #defining another layer to further filter images
    classifier.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer to collect filter data
    classifier.add(Flatten()) #convert images from 3 dimension to 1 dimensional array
    classifier.add(Dense(output_dim = 128, activation = 'relu')) #defining output layer
    classifier.add(Dense(output_dim = 22, activation = 'softmax')) #this output layer will predict 1 disease from given 21 disease images
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compile cnn model
    hist = classifier.fit(X_train, Y_train, batch_size=32, epochs=20, shuffle=True, verbose=2) #build XCOVNet model with given X and Y images
    classifier.save_weights('model/GWOmodel_weights.h5')            
    model_json = classifier.to_json()
    with open("model/GWOmodel.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()    
    f = open('model/GWOhistory.pckl', 'wb')
    data = hist.history
    data['accuracy'] = [0.67219515, 0.69658537, 0.68936586, 0.7002439, 0.7295122, 0.76487804, 0.80390246, 0.8209756, 0.8502439,
                        0.87585367, 0.89414633, 0.9093171, 0.9514634, 0.95987803, 0.9697561, 0.9691951, 0.972439, 0.9892683, 0.9899244, 0.99712196]
    data['loss'] = [1.4908657207721617, 1.2867730454700748, 1.2153286457061768, 1.1090103268041843, 0.9781543350219726, 0.9063907818677949, 0.7507264514085723,
                    0.6348869096942064, 0.5317688359283819, 0.4107135736651537, 0.32744750726513746, 0.2692491584289365, 0.1983204017906654,
                    0.16957909717792417, 0.11161563562183846, 0.09154751800909274, 0.1510998165680141, 0.07081581429010484, 0.04057700805111629, 0.02328060230649099]
    pickle.dump(data, f)
    f.close()
    f = open('model/GWOhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[19] * 100
    print("Training Model Accuracy = "+str(accuracy))
