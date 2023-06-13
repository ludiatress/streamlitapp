import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

st.title('Credit Card Fraud Detection!')

# Load the dataset from the csv file using pandas
df=st.cache_data(pd.read_csv)('creditcard.csv')
df = df.sample(frac=0.1, random_state = 48)

# Print shape and description of the data
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ',df.shape)
    st.write('Data decription: \n',df.describe())
# Print valid and fraud transactions
fraud=df[df.Class==1]
valid=df[df.Class==0]
outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100
if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
    st.write('Fraud Cases: ',len(fraud))
    st.write('Valid Cases: ',len(valid))

#Obtaining X (features) and y (labels)
X=df.drop(['Class'], axis=1)
y=df.Class
#Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)
#Print shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)
#Import classification models and metric
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)

#Feature selection through feature importance
st.cache_data
def feature_sort(model,X_train,y_train):
    #feature selection
    mod=model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp

#Classifiers for feature importance
clf=['Extra Trees','Random Forest']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)
start_time = timeit.default_timer()
if mod_feature=='Extra Trees':
    model=etree
    importance=feature_sort(model,X_train,y_train)
elif mod_feature=='Random Forest':
    model=rforest
    importance=feature_sort(model,X_train,y_train)
elapsed = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))

#Plot of feature importance
st.set_option('deprecation.showPyplotGlobalUse', False)
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()
      
#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,classification_report

def compute_performance(model, X_train, y_train,X_test,y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',scores
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm  
    cr=classification_report(y_test, y_pred)
    'Classification Report: ',cr
    elapsed = timeit.default_timer() - start_time
    'Execution Time for performance computation: %.2f minutes'%(elapsed/60)

#Run different classification models with rectifiers
if st.sidebar.checkbox('Run a credit card fraud detection model'):
    
    alg=['Extra Trees','Random Forest']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)
    rectifier=['No Rectifier']
    imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier) 
    
    if classifier == 'Extra Trees':
        model=etree
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train, y_train,X_test,y_test)
    elif classifier == 'Random Forest':
        model=rforest
        if imb_rect=='No Rectifier':
            compute_performance(model,X_train,y_train,X_test,y_test) 
    
    
    