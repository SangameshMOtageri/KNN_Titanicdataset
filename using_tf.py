import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd

df=pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
df.fillna(0, inplace=True)
with pd.option_context('display.max_rows', None,'display.max_columns', None):
    print(df.head())

def handle_numeric_data(df):
    columns=df.columns.values
        
    for column in columns:

        text_to_digits_vals={}

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            unique_set=set(df[column].values.tolist())
            x=0
            for unique in unique_set:
                text_to_digits_vals[unique]=x
                x+=1

            df[column]=[text_to_digits_vals[c] for c in df[column]]

    return df

df=handle_numeric_data(df)
with pd.option_context('display.max_rows', None,'display.max_columns', None):
    print(df.head())

X=np.array(df.drop(['survived'],1).astype(float))
X=preprocessing.scale(X)#improved accuracy from 51% to 71%
Y=np.array(df['survived'])

clf=KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1, len(predict_me))
    prediction=clf.predict(predict_me)
    if prediction == Y[i]:
        correct+=1
print('Accuracy: ',(lambda:correct/len(X), lambda:1-correct/len(X))[correct/len(X)<0.5]()*100)







