# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
import joblib
import pickle

def AdaBoostPreprocessor(df, inference=False):
    df.loc[df["Gender"]=="Male", 'Gender'] = True
    df.loc[df["Gender"]=="Female", 'Gender'] = False
    df.loc[df["H_Cal_Consump"]=="yes", 'H_Cal_Consump'] = True
    df.loc[df["H_Cal_Consump"]=="no", 'H_Cal_Consump'] = False
    ListAlcoholConsump = ['no', 'Sometimes', 'Frequently', 'Always']
    df['Alcohol_Consump']=df['Alcohol_Consump'].apply(lambda x: ListAlcoholConsump.index(str(x)))
    df.loc[df["Smoking"]=="yes", 'Smoking'] = True
    df.loc[df["Smoking"]=="no", 'Smoking'] = False
    df['Food_Between_Meals']=df['Food_Between_Meals'].apply(lambda x: ListAlcoholConsump.index(str(x)))
    df.loc[df["Fam_Hist"]=="yes", 'Fam_Hist'] = True
    df.loc[df["Fam_Hist"]=="no", 'Fam_Hist'] = False
    df.loc[df["H_Cal_Burn"]=="yes", 'H_Cal_Burn'] = True
    df.loc[df["H_Cal_Burn"]=="no", 'H_Cal_Burn'] = False
    ListTransport = ['Public_Transportation', 'Automobile', 'Walking', 'Bike', 'Motorbike']
    df['Transport'] = df['Transport'].apply(lambda x: ListTransport.index(str(x)))

    if(not inference):
      ListBodyLevel = ['Body Level 1', 'Body Level 2', 'Body Level 3', 'Body Level 4']
      df['Body_Level'] = df['Body_Level'].apply(lambda x: ListBodyLevel.index(str(x)))
      y=df['Body_Level']

    bmi=df['Weight']/(df['Height']**2)
    df['bmi'] = bmi
    y0=bmi*0
    y0[bmi>29.9]=4
    y0[bmi<=29.9]=3
    y0[bmi<=24.9]=2
    y0[bmi<18.5]=1
    return df

def InferAdaBoostedRF(df, path):
  df1 = AdaBoostPreprocessor(df, inference=True).copy()
  with open(path,'rb') as f:
    models = pickle.load(f)
  ssc = models[0]
  ada = models[1]
  df2 = ssc.transform(df1).copy()
  prediction = ada.predict(df2)
  ListBodyLevel = ['Body Level 1', 'Body Level 2', 'Body Level 3', 'Body Level 4']
  final = []
  for i in prediction:
    final.append(ListBodyLevel[int(i)])
  return final






test = pd.read_csv('test.csv')

predicted = InferAdaBoostedRF(test , 'model.pkl')

#TODO check how to write the preds.txt, will it be stored comma separated, one output per line, space separated, etc.?
with open('preds.txt', 'w') as f:
    f.write(predicted)