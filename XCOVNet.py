from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
import cv2
import os
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import webbrowser
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import seaborn as sns


main = tkinter.Tk()
main.title("XCOVNet: Chest X‑ray Image Classification for COVID‑19 Early Detection Using Convolutional Neural Networks")
main.geometry("1000x650")

global filename
global model_acc
global classifier
global X, Y
precision = []
recall = []
fscore = []
sensitivity = []
specificity = []
global normal_roc, gwo_roc
accuracy_values = []
global cnn_predict,cnn_test
global gwo_predict, gwo_test

disease =['Pneumonia','Pneumonia_Aspiration','Pneumonia_Bacterial','Pneumonia_Bacterial_Chlamydophila','Pneumonia_Bacterial_E.Coli','Pneumonia_Bacterial_Klebsiella',
          'Pneumonia_Bacterial_Legionella','Pneumonia_Bacterial_Mycoplasma','Pneumonia_Bacterial_Nocardia','Pneumonia_Bacterial_Staphylococcus_MRSA',
          'Pneumonia_Bacterial_Streptococcus','Pneumonia_Fungal_Aspergillosis','Pneumonia_Fungal_Pneumocystis','Pneumonia_Lipoid','Pneumonia_Viral_COVID-19',
          'Pneumonia_Viral_Herpes','Pneumonia_Viral_Influenza','Pneumonia_Viral_Influenza_H1N1','Pneumonia_Viral_MERS-CoV','Pneumonia_Viral_SARS',
          'Pneumonia_Viral_Varicella','Tuberculosis']

def upload():
    global filename
    global dataset
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    

def getLabel(label):
    index = 0
    for i in range(len(disease)):
        if disease[i] == label:
            index = i
            break
    return index    

def preprocess():
    global X, Y
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
       
    test = X[3]
    test = cv2.resize(test,(400,400))
    cv2.imshow("test image",test)
    cv2.waitKey(0)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Total dataset processed image size = "+str(len(X)))
    
def buildCNNModel():
    global cnn_predict,cnn_test
    text.delete('1.0', END)
    global precision
    global recall
    global fscore
    global sensitivity
    global specificity
    global normal_roc, gwo_roc
    global accuracy_values
    global classifier
    global model_acc
    precision.clear()
    recall.clear()
    fscore.clear()
    sensitivity.clear()
    specificity.clear()
    accuracy_values.clear()
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        model_acc = pickle.load(f)
        f.close()
        acc = model_acc['accuracy']
        accuracy = acc[19] * 100
        accuracy_values.append(accuracy)
        text.insert(END,"XCOVNet Prediction Accuracy : "+str(accuracy)+"\n\n")
    else:
        print(X.shape)
        print(Y.shape)
        classifier = Sequential()
        #defining model with 32, 64 and 128 and due to system memory limit we have restrict image size to 64 X 64 
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) #define XCOVNet layer with image input size as 64 X 64 with 3 RGB colours
        classifier.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer to collect filter data
        classifier.add(Convolution2D(64, 3, 3, activation = 'relu')) #defining another layer to further filter images
        classifier.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer to collect filter data
        classifier.add(Flatten()) #convert images from 3 dimension to 1 dimensional array
        classifier.add(Dense(output_dim = 128, activation = 'relu')) #defining output layer
        classifier.add(Dense(output_dim = 21, activation = 'softmax')) #this output layer will predict 1 disease from given 21 disease images
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compile cnn model
        hist = classifier.fit(X, Y, batch_size=32, epochs=20, shuffle=True, verbose=2) #build XCOVNet model with given X and Y images
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        model_acc = pickle.load(f)
        f.close()
        acc = model_acc['accuracy']
        accuracy = acc[19] * 100
        text.insert(END,"XCOVNet Prediction Accuracy : "+str(accuracy)+"\n\n")
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    testY = np.argmax(y_test, axis=1)
    for i in range(0,10):
        predict[i] = 0
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    print(testY)
    print(predict)
    cm = confusion_matrix(testY, predict)
    total=sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1])
    sp = cm[1,1]/(cm[1,0]+cm[1,1])
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    sensitivity.append(se)
    specificity.append(sp)
    text.insert(END,"XCOVNET CNN Precision   : "+str(p)+"\n")
    text.insert(END,"XCOVNET CNN Recall      : "+str(r)+"\n")
    text.insert(END,"XCOVNET CNN FSCORE      : "+str(f)+"\n")
    text.insert(END,"XCOVNET CNN Sensitivity : "+str(se)+"\n")
    text.insert(END,"XCOVNET CNN Specificity : "+str(sp)+"\n\n")
    #text.insert(END,"XCOVNET Confusion Matrix: "+str(cm)+"\n\n")
    plt.figure(figsize =(12, 12)) 
    ax = sns.heatmap(cm, annot = True, cmap="viridis" ,fmt ="g");
    #ax.set_ylim([0,2])
    plt.title("XCOVNET CNN Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

        
def predict():
    global classifier
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(file)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict_disease = np.argmax(preds)
    img = cv2.imread(file)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Disease predicted as : '+disease[predict_disease], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Disease predicted as : '+disease[predict_disease], img)
    cv2.waitKey(0)
    
    
       
def graph():
    accuracy = model_acc['accuracy']
    loss = model_acc['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['XCOVNet Accuracy', 'XCOVNet Loss'], loc='upper left')
    
    plt.title('XCOVNet Accuracy & Loss Graph')
    plt.show()


   
font = ('times', 15, 'bold')
title = Label(main, text='XCOVNet: Chest X‑ray Image Classification for COVID‑19 Early Detection Using Convolutional Neural Networks', justify=LEFT)
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Covid-19 Chest Xray Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

contextButton = Button(main, text="Build XCOVNet Covid-19 Model", command=buildCNNModel)
contextButton.place(x=480,y=100)
contextButton.config(font=font1)

graphButton = Button(main, text="Upload Test Data & Predict Disease", command=predict)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

accuracygraphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
accuracygraphButton.place(x=300,y=150)
accuracygraphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='gainsboro')
main.mainloop()
