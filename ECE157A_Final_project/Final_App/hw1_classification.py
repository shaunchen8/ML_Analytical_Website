import mpld3
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.model_selection import cross_validate as cv
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cmap

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

def run_algo1(file_path):

    c,d = load_data('Final_App/diabetes.csv')
    c,xs,ys = separate_labels(c,d)
    #columnNames, xs = preprocess_data(c,xs)

    xtrain,xtest,ytrain,ytest = train_test_split(xs,ys,train_size=0.8,random_state=42)

    clf = LDA()
    clf.fit(xtrain,ytrain)

    ##scores
    #cv_result = cv(clf,xtrain,ytrain,return_train_score=True)
    #print('cross validation test score:', cv_result['test_score'])
    #print('cross validation train score:', cv_result['train_score'])
    #print('score:', clf.score(xtest,ytest))

    ##validation results and confusion matrix
    pred_y = clf.predict(xtest)
    #print('Validation Results')
    #print(cm(ytest,pred_y))
    #print(cr(ytest,pred_y))

    feature0 = 0 #pregnancies
    feature1 = 1 #glucose
    feature2 = 2 #blood pressure
    feature3 = 3 #skinthickness
    feature4 = 4 #insulin 
    feature5 = 5 #BMI 
    feature6 = 6 #diabetes pedigree func
    feature7 = 7 #age

    columnNames, rows = load_data(file_path)
    #columnNames, rows = preprocess_data(columnNames,rows)
    test_x = rows
    #print(test_x.shape)
    pred_ytest = clf.predict(test_x)
    pred_ytest = list(pred_ytest)

    fig = plt.figure()
    scatter = plt.scatter(test_x[:,feature1],test_x[:,feature5],10,pred_ytest,cmap='jet')
    plt.title(label='Scatter of Diabetic Patients using the LDA Classifier')
    plt.xlabel(c[feature1])
    plt.ylabel(c[feature5])
    handles, labels = scatter.legend_elements()
    plt.legend(handles=handles,labels=['Nondiabetic','Diabetic'], loc="upper right", title="Legend")
    #plt.show()

    return mpld3.fig_to_html(fig)

#run_algo1('D://Documents - Data Drive/VSCode/ECE157A_Final_Project/Final_Project/Final_App/hw1_unknowns_diabetes.csv')
#plt.show()   