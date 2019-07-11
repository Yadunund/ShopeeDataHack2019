#!/usr/bin/env python
# coding: utf-8

# In[1]:

__author__      = "Yadunund Vijay"
"""
Python Executable to train random forest model to rpredict product labels from text and image inputs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import confusion_matrix, classification_report
#from xgboost import XGBClassifier

def text_process(mess):
    """
    1. Remove punc
    2. Remove stop words 
    3. Return list of clean text words    
    """
    nopunc= [char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english') and len(word)>2]
    

from sklearn.externals import joblib
def traindata(dataset,train_all=True,attribute='Benefits',n_estimators=10,use_imagetext=False,save_model=True):
    paths_train=['fashion_data_info_train_competition.csv','beauty_data_info_train_competition.csv','mobile_data_info_train_competition.csv']
    paths_val=['fashion_data_info_val_competition.csv','beauty_data_info_val_competition.csv','mobile_data_info_val_competition.csv']
    paths_json=['fashion_profile_train.json','beauty_profile_train.json','mobile_profile_train.json']
    path_train=''
    path_val=''
    path_json=''

    if(dataset=='fashion'):
        filename=paths_train[0]
    elif(dataset=='beauty'):
        filename=paths_train[1]
    else:
        filename=paths_train[2]


    print('Path_train:'+filename)
    
    df= pd.read_csv(filename)
    df.fillna(-1,inplace=True)
    
    if(use_imagetext):
        try:
            tmp=filename.split('.')
            filename_t=tmp[0]+'_imtext.csv'
            tdf=pd.read_csv(filename_t)
            df['title']=df['title']+' ' +tdf['title_image']
        except e:
            print('Error in loading imtext file. Skipping...')
    X=df['title']
    df_filt=df.iloc[:,3:]
    accuracy=[]
    if(train_all):
        for column in df_filt:
            y=df_filt[column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            p=Pipeline([
                    ('bow',CountVectorizer(analyzer=text_process)),
                ('tfidf',TfidfTransformer()),
                ('classifier',RandomForestClassifier(n_estimators=n_estimators,verbose=2,n_jobs=-1))  
                ])
            print("Training attribute "+ str(column)+ " with "+str(n_estimators)+ " estimators")

            p.fit(X_train,y_train)
            pred=p.predict(X_test)
            conf_mat = confusion_matrix(y_test,pred)
            a=np.sum(np.diagonal(conf_mat))/np.sum(conf_mat)
            print(column)
            print("Accuracy: ", a)
            accuracy.append(a)

            # save the model to disk
            modelname =dataset+'_'+'rf_'+'n_'+str(n_estimators)+column + '.sav'
            print("Saving mode to file "+ modelname)
            if(save_model):
                #pickle.dump(p, open(modelname, 'wb'))
                joblib.dump(p,modelname)
    else:
        column=attribute
        y=df_filt[column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        p=Pipeline([
                ('bow',CountVectorizer(analyzer=text_process)),
            ('tfidf',TfidfTransformer()),
            ('classifier',RandomForestClassifier(n_estimators=n_estimators,verbose=2,n_jobs=-1))  
            ])
        #XGBClassifier(n_estimators=n_estimators,verbosity =2)
        print("Training attribute "+ str(column)+ " with "+str(n_estimators)+ " estimators")

        p.fit(X_train,y_train)
        pred=p.predict(X_test)
        conf_mat = confusion_matrix(y_test,pred)
        a=np.sum(np.diagonal(conf_mat))/np.sum(conf_mat)
        print(column)
        print("Accuracy: ", a)
        accuracy.append(a)

        # save the model to disk
        modelname =dataset+'_'+'rf_'+'n_'+str(n_estimators)+column + '.sav'
        print("Saving mode to file "+ modelname)
        if(save_model):
            #pickle.dump(p, open(modelname, 'wb'))
            joblib.dump(p,modelname)

    
    return([df_filt.columns,accuracy])

traindata('mobile',train_all=False,attribute='Phone Model',n_estimators=10,use_imagetext=False,save_model=True)
#traindata('mobile',train_all=False,attribute='Phone Screen Size',n_estimators=50,use_imagetext=False,save_model=True)


