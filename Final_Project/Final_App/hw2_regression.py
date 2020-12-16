import mpld3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    with open(filename,'r') as file:
        reader = csv.reader(file)
        columnNames = next(reader)
        rows = np.array(list(reader), dtype=float)
        return columnNames, rows

def separate_labels(columnNames, rows):
    labelColumnIndex = columnNames.index('Outcome')
    ys = rows[:, labelColumnIndex]
    xs = np.delete(rows,labelColumnIndex,axis=1)
    del columnNames[labelColumnIndex]
    return columnNames, xs, ys

def train_test_split_two(xs,ys):
    train_x3, test_x3, train_y3, test_y3 = train_test_split(xs[ys==3],ys[ys==3],test_size=0.5,random_state=42)
    train_x4, test_x4, train_y4, test_y4 = train_test_split(xs[ys==4],ys[ys==4],test_size=0.5,random_state=42)
    condition567 = np.logical_or(np.logical_or(ys==5,ys==6),ys==7)
    train_x567, test_x567, train_y567, test_y567 = train_test_split(xs[condition567],ys[condition567],test_size=0.66,random_state=42)
    train_x8, test_x8, train_y8, test_y8 = train_test_split(xs[ys==8],ys[ys==8],test_size=0.5,random_state=42)
    train_x9, test_x9, train_y9, test_y9 = train_test_split(xs[ys==9],ys[ys==9],test_size=0.5,random_state=42)
    train_x = np.concatenate([train_x3,train_x4,train_x567,train_x8,train_x9])
    test_x = np.concatenate([test_x3,test_x4,test_x567,test_x8,test_x9])
    train_y = np.concatenate([train_y3,train_y4,train_y567,train_y8,train_y9])
    test_y = np.concatenate([test_y3,test_y4,test_y567,test_y8,test_y9])
    return train_x, test_x, train_y, test_y

def run_algo2(file_path):

    df = pd.read_csv('Final_App/white_wine.csv')
    #df = pd.read_csv('../media/upload/white_wine.csv')
    ys = df['quality']
    xs = df.drop('quality',1)

    train_x, test_x, train_y, test_y = train_test_split_two(xs,ys)
    #print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)

    clf=RFR(n_estimators=100,max_depth=20,random_state=0).fit(train_x,train_y)

    pred_y=np.round(clf.predict(test_x))
    #print(accuracy_score(test_y,pred_y))
    #print(explained_variance_score(test_y,pred_y))
    #plt.scatter(test_y,pred_y)
    #plt.show()

    #from sklearn.model_selection import cross_validate as cv
    #cv_result = cv(clf,train_x,train_y,return_train_score=True)
    #print('cross validation test score:', cv_result['test_score'])
    #print('cross validation train score:', cv_result['train_score'])
    #print('score:', clf.score(test_x,test_y))

    #print('ValidationResults')
    #print(cm(test_y,pred_y))
    #print(cr(test_y,pred_y))

    fig = plt.figure()
    feature9 = 9 #sulphates
    feature10 = 10 #alcohol
    scatter = plt.scatter(test_x[:,feature9],test_x[:,feature10],10,pred_y, cmap='jet')
    plt.title(label="Scatter of White Wine Data using the Random Forest Regressor")
    plt.xlabel('sulphates')
    plt.ylabel('alcohol')
    #handles, labels = scatter.legend_elements()
    #plt.legend(handles=handles,labels=['Score = 4','Score = 5','Score = 6','Score = 7','Score = 8'], loc="upper right", title="Legend")

    fig.colorbar(scatter)
    #plt.show()

    return mpld3.fig_to_html(fig)

#run_algo2('unknowns_white_wine.csv')
#plt.show()   