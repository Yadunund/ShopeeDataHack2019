#!/usr/bin/env python
# coding: utf-8

"""
To extract text information from gigabytes of product photos using OCR
"""
__author__      = "Yadunund Vijay"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import matplotlib.patches as patches
import pickle

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import argparse
import os

import cv2



#Read data into dataframe df
df=pd.read_csv('beauty_data_info_train_competition.csv')
df.fillna(-1,inplace=True)

#add subfolder name to path directory
pathadd='beautyimages/' # path to be appended to the image path in df['image_path']
df['image_path']=pathadd+df['image_path']

def gettext(path):
    text=''
    try:
        text=pytesseract.image_to_string(Image.open(path))
        text=' '.join(text.split())
    except:
        text=''
    print('___________'.encode("utf-8"))
    print(text.encode("utf-8"))
    print('___________'.encode("utf-8"))
    return text
            
def displayimage(filename):
    im = np.array(Image.open(filename), dtype=np.uint8)
    # Create figure and axes
    fig,ax = plt.subplots(1,figsize=(8,6))
    # Display the image
    ax.imshow(im)


def gettext_cv(path,mode='ADAPTIVE_GAUSSIAN',c1=13,c2=12):
    '''
    Author:Yadunund Vijay
    Docstring: Function to use OCR to read text from image with option of
                applying thresholding on image based on user supplied 'mode' and constants 'c1','c2'.
                modes: {'Normal': for no thresholding, 'ADAPTIVE_GAUSSIAN','ADAPTIVE_MEAN','BINARY','BINARY_OTSU'
    '''
    text=''
    try:
        if(mode=='NORMAL'):
            text=pytesseract.image_to_string(Image.open(path))
        else:
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            if (mode=='ADAPTIVE_GAUSSIAN'):
                    gray=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, c1, c2)
            if (mode=='ADAPTIVE_MEAN'):
                    gray=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, c1, c2)
            if(mode=='BINARY_OTSU'):
                    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            if(mode=='BINARY'):
                    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY)[1]
            
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)
        text = pytesseract.image_to_string(Image.open(filename))
        text=' '.join(text.split())
        os.remove(filename)
        # Display the image
        fig,ax = plt.subplots(ncols=1,figsize=(8,6))
        ax.imshow(image)
    except:
        text=''
    print('___________')
    print(text)
    print('___________')
    return text


# # Calling function to add new column to dataframe containing text from images
df['title_image']=df['image_path'].apply(gettext)
df.to_csv('beauty_data_info_train_competition_imtext.csv',index=False)


