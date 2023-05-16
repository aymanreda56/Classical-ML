import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from sklearn.pipeline import make_pipeline
import joblib
import pickle
import matplotlib.pyplot as plt
from mlxtend.evaluate import bias_variance_decomp




def AdaBoostPreprocessor(df, inference=False, add_Custom_Features=True):
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
    #df['Transport'] = LabelEncoder().fit_transform(df['Transport'])
    ListTransport = ['Public_Transportation', 'Automobile', 'Walking', 'Bike', 'Motorbike']
    df['Transport'] = df['Transport'].apply(lambda x: ListTransport.index(str(x)))

    if(not inference):
      #df['Body_Level'] = LabelEncoder().fit_transform(df['Body_Level'])
      ListBodyLevel = ['Body Level 1', 'Body Level 2', 'Body Level 3', 'Body Level 4']
      df['Body_Level'] = df['Body_Level'].apply(lambda x: ListBodyLevel.index(str(x)) + 1)
      y=df['Body_Level']

    if(add_Custom_Features):
      bmi=df['Weight']/(df['Height']**2)
      df['bmi'] = bmi
      y0=bmi*0
      y0[bmi>29.9]=4
      y0[bmi<=29.9]=3
      y0[bmi<=24.9]=2
      y0[bmi<18.5]=1

    return df
    
def AdaBoostSMOTE(df):
    oversample = SMOTE()
    y=df['Body_Level']
    X=df.loc[:, df.columns != 'Body_Level']

    X0, y = oversample.fit_resample(X, y)
    df = X0
    df['Body_Level'] = y
    return df