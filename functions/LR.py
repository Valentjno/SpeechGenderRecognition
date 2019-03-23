#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:29:29 2019

@author: valentjno, helias
"""

import numpy
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer

def normalize_L2(X_train,X_test):    
    norm = Normalizer(norm='l2')
    X_training_l2 = norm.transform(X_train)
    X_test_l2 = norm.transform(X_test)
    return X_training_l2,X_test_l2

#normalizzed L2.
def PCA_decomposition(X_train,X_test):
    from sklearn.decomposition import PCA
    pca=PCA()
    pca.fit(X_train)
    X_train_pca=pca.transform(X_train)
    X_test_pca=pca.transform(X_test)
    return X_train_pca,X_test_pca, pca

def fit_LR(X_train,y_train):
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()
    lr.fit(X_train,y_train.ravel())
    return lr

def fit_Bernoulli_NB(X_train,y_train):
    from sklearn.naive_bayes import BernoulliNB
    nb=BernoulliNB()
    nb.fit(X_train,y_train.ravel())
    return nb

'''
The possible parameters are:
  _algorithm:
      - "ball_tree","kd_tree","brute","auto".
  _weights:
      - "uniform"(default), "distance".
'''
def fit_KNN(X_train,y_train,_algorithm="",_weights="uniform"):
    from sklearn import neighbors
    if(_algorithm==""):
        clf = neighbors.KNeighborsClassifier(11, weights=_weights)
    else:        
        clf = neighbors.KNeighborsClassifier(11, algorithm=_algorithm, weights=_weights)
    clf.fit(X_train, y_train.ravel())
    return clf

# _gamma= "auto" or "scale"
def fit_SVM(X_train,y_train, _gamma="auto"):
    from sklearn import svm
    clf = svm.NuSVC(gamma=_gamma)
    clf.fit(X_train, y_train.ravel())
    return clf

def plot_2D(lr,X_train,y_train):
    x_f=X_train[:,0]
    y_f=X_train[:,1]
    plt.figure()    
    plt.plot(x_f[y_train==0],y_f[y_train==0],"or")
    plt.plot(x_f[y_train==1],y_f[y_train==1],"og")
    thetaN=lr.coef_
    theta0=lr.intercept_
    theta1=thetaN[0][0]
    theta2=thetaN[0][1]
    x=numpy.array([-0.9,0.9])
    y=-((theta0+theta1)*x)/(theta2)
    plt.plot(x,y)
    plt.show()

def plot_3D(lr,X_train,y_train):    
    from mpl_toolkits.mplot3d import Axes3D
    plt.figure()
    plt.subplot(111,projection="3d")
    x_f=X_train[:,0]
    y_f=X_train[:,1]
    z_f=X_train[:,2]
    plt.plot(x_f[y_train==0],x_f[y_train==0],z_f[y_train==0],"or")
    plt.plot(x_f[y_train==1],y_f[y_train==1],z_f[y_train==1],"og")
    thetaN=lr.coef_
    theta0=lr.intercept_
    theta1=thetaN[0][0]
    theta2=thetaN[0][1]
    theta3=thetaN[0][2]
    x=numpy.array([-0.9,0.9])
    y=numpy.arange(-0.9,0.9)
    x,y=numpy.meshgrid(x,y)
    z=-(theta0+theta1*x+theta2*y)/(theta3)
    plt.gca().plot_surface(x,y,z,shade=False,color='y')
    plt.show()
    
def predict_and_score(lr,x_test,y_test):
    from sklearn.metrics import accuracy_score, confusion_matrix
    p_test=lr.predict(x_test)
    return accuracy_score(y_test,p_test), confusion_matrix(y_test,p_test)

def run_classifier(x_train, y_train, x_test, y_test):
    lr = fit_LR(x_train, y_train)
    acc, conf_matrix = predict_and_score(lr, x_test, y_test)
    print("LR: \t", acc)

    nb = fit_Bernoulli_NB(x_train, y_train)
    acc, conf_matrix = predict_and_score(nb, x_test, y_test)
    print("NB: \t", acc)

    svm = fit_SVM(x_train, y_train, _gamma="scale")
    acc, conf_matrix = predict_and_score(svm, x_test, y_test)
    print("SVM: \t", acc)

    _knn = fit_KNN(x_train, y_train, _algorithm="ball_tree", _weights="distance")
    acc, conf_matrix = predict_and_score(_knn, x_test, y_test)
    print("KNN: \t", acc)
    
    save_model(lr,"models_trained/lr_model")
    save_model(nb,"models_trained/nb_model")
    save_model(svm,"models_trained/svm_model")
    save_model(_knn,"models_trained/knn_model")

#after the fit
def save_model(model,filename):
    _filename=filename+'.sav'
    pickle.dump(model, open(_filename, 'wb'))
    
def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
