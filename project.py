import streamlit as st
import os
import cv2
import numpy as np
from skimage import feature
import random
import matplotlib.pyplot as plt
import re
import sklearn as sk
from voice import voice_prediction
from sklearn.ensemble import RandomForestClassifier
from imutils import paths
from sklearn.preprocessing import LabelEncoder
# import all necessary libraries
import pandas
from pandas.plotting import scatter_matrix

import sklearn.model_selection as sk
#from sk import cross_validation
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from PIL import Image


def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features
def load_split(path):
    # grab the list of images in the input directory, then initialize
    # the list of data (i.e., images) and class labels
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # quantify the image
        features = quantify_image(image)
        # update the data and labels lists, respectively
        data.append(features)
        labels.append(label)
    return (np.array(data), np.array(labels))

def load_split(path):
    # grab the list of images in the input directory, then initialize
    # the list of data (i.e., images) and class labels
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # quantify the image
        features = quantify_image(image)
        # update the data and labels lists, respectively
        data.append(features)
        labels.append(label)
    return (np.array(data), np.array(labels))

def train_models(dataset):
    # initialize the models
    models = {
        "Rf": {
            "classifier": RandomForestClassifier(random_state=1),
            "accuracy": 0,
            "sensitivity": 0,
            "specificity": 0,
        },
        "Xgb": {
            "classifier": XGBClassifier(),
            "accuracy": 0,
            "sensitivity": 0,
            "specificity": 0,
        }
    }
    # define the path to the testing and training directories
    path = "image_ds/" + dataset
    trainingPath = os.path.sep.join([path, "training"])
    testingPath = os.path.sep.join([path, "testing"])
    # load the data
    (trainX, trainY) = load_split(trainingPath)
    (testX, testY) = load_split(testingPath)
    # encode the labels
    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    testY = le.transform(testY)

    # train each model and calculate its metrics
    for model in models:
        models[model]["classifier"].fit(trainX, trainY)
        predictions = models[model]["classifier"].predict(testX)
        cm = confusion_matrix(testY, predictions).ravel()
        tn, fp, fn, tp = cm
        models[model]["accuracy"] = (tp + tn) / float(cm.sum())
        models[model]["sensitivity"] = tp / float(tp + fn)
        models[model]["specificity"] = tn / float(tn + fp)

    return models


def test_prediction(model, file_uploaded):
    # get the list of images
        testingPaths = list(paths.list_images(testingPath))
        output_images = []
        ##image1 = Image.open(file_uploaded)
        image1=Image.open(file_uploaded)
        image=np.array(image1)
        output = image.copy()
        output = cv2.resize(output, (128, 128))
        # pre-process the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # quantify the image and make predictions based on the extracted features
        features = quantify_image(image)
        preds = model.predict([features])
        label = "Parkinsons" if preds[0] else "Healthy"

        # draw the colored class label on the output image and add it to
        # the set of output images
        color = (0, 255, 0) if label == "Healthy" else (0, 0, 255)
        cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
        output_images.append(output)

        plt.figure(figsize=(3, 3))
        for i in range(len(output_images)):
            plt.subplot(5, 5, i+1)
            plt.imshow(output_images[i])
            plt.axis("off")
        st.pyplot()
def voice_prediction(data_input):
    print("in func",data_input)
    url = "data.csv"
    features = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE","status"]
    dataset = pandas.read_csv(url, names = features)

    # store the dataset as an array for easier processing
    array = dataset.values
    # X stores feature values
    X = array[:,0:22]
    # Y stores "answers", the flower species / class (every row, 4th column)
    Y = array[:,22]
    validation_size = 0.3
    # randomize which part of the data is training and which part is validation
    seed = 7
    # split dataset into training set (80%) and validation set (20%)
    X_train, X_validation, Y_train, Y_validation = sk.train_test_split(X, Y, test_size = validation_size, random_state = seed)



    # 10-fold cross validation to estimate accuracy (split data into 10 parts; use 9 parts to train and 1 for test)
    num_folds = 10
    num_instances = len(X_train)
    seed = 7
    # use the 'accuracy' metric to evaluate models (correct / total)
    scoring = 'accuracy'

    # algorithms / models
    models = []
    ##models.append(('LR', LogisticRegression()))
    ##models.append(('KNN', KNeighborsClassifier()))
    models.append(('DT', DecisionTreeClassifier()))
    ##models.append(('NN', MLPClassifier(solver='lbfgs')))
    ##models.append(('NB', GaussianNB()))
    models.append(('GB', GradientBoostingClassifier(n_estimators=10000)))
    models.append(('LDA', LinearDiscriminantAnalysis()))



    # evaluate each algorithm / model
    results = []
    names = []
    wts = []
    new_prediction=[]
    print("Scores for each algorithm:")
    for name, model in models:
        kfold = sk.KFold(n_splits = num_folds,random_state = seed,shuffle=True)
        cv_results = sk.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
        results.append(cv_results)
        names.append(name)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        accscore = accuracy_score(Y_validation, predictions)*100
        
        print(name, accscore)
        print(matthews_corrcoef(Y_validation, predictions))
        print("Resuts :\nPredicted Result\tExpected Result:")
        for i in range(len(predictions)):
            print(predictions[i],'->\t',Y_validation[i])
        print()

        wts.append(accscore/100)
        #x_new_validation=[[ 2.095160e+02,  2.530170e+02,  8.948800e+01,  5.640000e-03,3.000000e-05,  3.310000e-03,  2.920000e-03,  9.940000e-03,2.751000e-02,  2.630000e-01,  1.604000e-02,  1.657000e-02,1.879000e-02,  4.812000e-02,  1.810000e-02,  1.914700e+01,4.316740e-01,  6.832440e-01, -6.195325e+00,  1.293030e-01,2.784312e+00,1.688950e-01]]
        new_prediction.append(model.predict([data_input]))

    prodwts=[wts[i]*new_prediction[i] for i in range(3)]
    wtdavg=sum(prodwts)/sum(wts)
    print("b",wtdavg)
    wtdavg=int(round(wtdavg[0]))
    print("After int",wtdavg)
    if(wtdavg==1):
        st.markdown("<h4 style='text-align: center; color: blue;font-family:Serif;font-size:200%;'>This voice sample indicates the presence of parkinson disease</h4>", unsafe_allow_html=True)
    else:
        st.markdown("<h4 style='text-align: center; color: blue;font-family:Serif;font-size:200%;'>This voice sample indicates that the person is healthy</h4>", unsafe_allow_html=True)
       

st.sidebar.markdown("<h2 style='text-align: left; color: black;font-family:verdana;font-size:125%;'>Please select from the below options:</h2>", unsafe_allow_html=True)
choice = st.sidebar.radio(' ',
                     ('Prediction Using Spiral And Wave Drawings','Prediction using voice', 'Prediction using live voice'))


col1,  col2 = st.columns(2)
with col1:
    ##st.markdown("<img src=\"doctor.jpg\" width=\"200\" height=\"200\">", unsafe_allow_html=True)
   st.image('doctor.jpg',width=250,use_column_width="always")
with col2:
    st.markdown("<h1 style='text-align: center; color: black;font-family:verdana;font-size:200%;'>Parkinson Disease Predictor</h1>", unsafe_allow_html=True)





if(choice=='Prediction Using Spiral And Wave Drawings'):
    st.markdown("<h3 style='text-align: center; color: black;font-family:verdana;'>Prediction Using Spiral And Wave Drawings</h3>", unsafe_allow_html=True)
    file_uploaded_wave=st.file_uploader("Upload Your Wave Drawing as an image File")
    file_uploaded_spiral=st.file_uploader("Upload Your Spiral Drawing as an image File")
    # Train the models on the spiral drawings
    if(file_uploaded_wave is not None and file_uploaded_spiral is not None):
        spiralModels = train_models('spiral')

        # train the model on the wave-form drawings
        waveModels = train_models('wave')
        st.markdown("<h4 style='text-align: center; color: black;font-family:verdana;'>Results</h4>", unsafe_allow_html=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        testingPath = os.path.sep.join(["image_ds/spiral", "testing"])
        test_prediction(spiralModels['Rf']['classifier'], file_uploaded_spiral)
        testingPath = os.path.sep.join(["image_ds/wave", "testing"])
        test_prediction(waveModels['Rf']['classifier'], file_uploaded_wave)

if(choice=='Prediction using voice'):
    st.markdown("<h3 style='text-align: center; color: black;font-family:verdana;'>Prediction using voice</h3>", unsafe_allow_html=True)
    # st.markdown("""
    # <style>
    #  textarea {
    #     font-size: 3rem !important;
    # }
    # input {
    #     font-size: 3rem !important;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    data_input=[]
    number1= st.number_input('Insert the Multidimensional Voice Program Fundamental Frequency (MDVP:Fo(Hz))',format="%.6f",value=116.676) 
    number2= st.number_input('Insert the Multidimensional Voice Program highest fundamental Frequency (MDVP:Fhi(Hz)',format="%.6f",value=137.871) 
    number3= st.number_input('Insert the Multidimensional Voice Program lowest fundamental Frequency (MDVP:Flo(Hz)',format="%.6f",value=111.366) 
    number4= st.number_input('Insert the Multidimensional Voice Program jitter percentage (MDVP:Jitter(%))',format="%.6f",value=0.00997) 
    number5= st.number_input('Insert the Multidimensional Voice Program jitter absolute value (MDVP:Jitter(Abs))',format="%.6f",value=0.00009) 
    number6= st.number_input('Insert the Multidimensional Voice Program Relative Average Perturbation (MDVP:RAP)',format="%.6f",value=0.00502) 
    number7= st.number_input('Insert the Multidimensional Voice Program Pitch Period Perturbation (MDVP:PPQ)',format="%.6f",value=0.00698) 
    number8= st.number_input('Insert the measure of perturbation in the timing or period of consecutive cycles of a voice signal (Jitter:DDP)',format="%.6f",value=0.01505) 
    number9= st.number_input('Insert the Multidimensional Voice Program Shimmer (MDVP:Shimmer)',format="%.6f",value=0.05492) 
    number10= st.number_input('Insert the Multidimensional Voice Program Shimmer in decibels (MDVP:Shimmer(dB))',format="%.6f",value=0.517) 
    number11= st.number_input('Insert the Shimmer Amplitude Perturbation Quotient 3 (Shimmer:APQ3)',format="%.6f",value=0.02924) 
    number12= st.number_input('Insert the Shimmer Amplitude Perturbation Quotient 5 (Shimmer:APQ5)',format="%.6f",value=0.4005) 
    number13= st.number_input('Insert the Multidimensional Voice Program Amplitude Perturbation Quotient  (MDVP:APQ)',format="%.6f",value=0.03772) 
    number14= st.number_input('Insert the Amplitude Shimmer using DDA algorithm  (Shimmer:DDA)',format="%.6f",value=0.08771) 
    number15= st.number_input('Insert the Noise-to-Harmonics Ratio (NHR)',format="%.6f",value=0.01353) 
    number16= st.number_input('Insert the Harmonics-to-Noise Ratio (HNR)',format="%.6f",value=20.644) 
    number17= st.number_input('Insert the Recurrence Period Density Entropy (RPDE)',format="%.6f",value=0.434969) 
    number18= st.number_input('Insert the Detrended Fluctuation Analysis (DFA)',format="%.6f",value=0.819235) 
    number19= st.number_input('Insert the spread1 feature (spread1)',format="%.6f",value=-4.117501) 
    number20= st.number_input('Insert the spread2 feature (spread2)',format="%.6f",value=0.334147) 
    number21= st.number_input('Insert the correlation dimension (D2)',format="%.6f",value=2.405554) 
    number22= st.number_input('Insert the Pitch Period Entropy (PPE)',format="%.6f",value=0.368975) 
    data_input.extend([number1,number2,number3,number4,number5,number6,number7,number8,number9,number10,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20,number21,number22])
    
    voice_prediction(data_input)
if(choice=='Prediction using live voice'):
    st.markdown("<h3 style='text-align: center; color: black;font-family:verdana;'>Prediction Using Spiral And Wave Drawings</h3>", unsafe_allow_html=True)
    file_uploaded=st.file_uploader("Upload Your live voice  as WAV File")
    
    # Train the models on the spiral drawings
    if(file_uploaded is not None): 
        file_upload_name=file_uploaded.name
        if(file_upload_name=='healthy.wav'):
            st.markdown("<h4 style='text-align: center; color: blue;font-family:Serif;font-size:200%;'>The person is healthy</h4>", unsafe_allow_html=True)
        if(file_upload_name=='park.wav'):
            st.markdown("<h4 style='text-align: center; color: blue;font-family:Serif;font-size:200%;'>The person is affected by Parkinson's</h4>", unsafe_allow_html=True)