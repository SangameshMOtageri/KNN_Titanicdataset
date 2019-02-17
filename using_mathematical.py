import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd

df=pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
df.fillna(0, inplace=True)
with pd.option_context('display.max_rows', None,'display.max_columns', None):
    print(df.head())

class KMEANS:
    def __init__(self,k=2,max_iter=50):
        self.k=k
        self.max_iter=max_iter
        self.centroid={}

    def fit(self,data):
        for i in range(self.k):
            self.centroid[i]=data[i]

        for iter in range(self.max_iter):
            print('Iteration: ',iter)
            self.classification={}
            for i in range(self.k):
                self.classification[i]=[]
            for i_data in data:

                distance=[np.average((i_data - self.centroid[c])**2) for c in range(len(self.centroid))]
                #print(min(distance))
                classified=distance.index(min(distance))
                #print(classified)
                self.classification[classified].append(i_data)

            for i in range(self.k):
                self.centroid[i]=np.average(self.classification[i], axis=0)

    def predict(self, data):

        distance=[np.average((data-self.centroid[c])**2) for c in range(len(self.centroid))]
        return distance.index(min(distance))
    
def handle_numeric_data(df):
    columns=df.columns.values
        
    for column in columns:

        text_to_digits_vals={}

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            unique_set=set(df[column].values.tolist())
            x=0
            for unique_i in unique_set:
                text_to_digits_vals[unique_i]=x
                x+=1

            df[column]=[text_to_digits_vals[c] for c in df[column]]

    return df
#some data is not in numerical form, so either convert to numerical or drop the
#feature
df=handle_numeric_data(df)
with pd.option_context('display.max_rows', None,'display.max_columns', None):
    print(df.head())

X=np.array(df.drop(['survived'],1).astype(float))
X=preprocessing.scale(X)#improved accuracy from 51% to 71%
#scaling make other features more visible 
Y=np.array(df['survived'])
clf=KMEANS()
clf.fit(X)

correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1, len(predict_me))
    prediction=clf.predict(predict_me)
    if prediction == Y[i]:
        correct+=1
print('Accuracy: ',(lambda:correct/len(X), lambda:1-correct/len(X))[correct/len(X)<0.5]()*100)


            




