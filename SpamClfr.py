#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:27:16 2018

@author: yty
"""
from sklearn.model_selection import KFold
import numpy as np


""" execuate different classifier and compare performance """
def cv(clsfr,dataset,k):
    pre = []
    kf = KFold(n_splits=k,shuffle= True)
    j = 1
    for train_index,test_index in kf.split(dataset[0]):
        trainset_x = dataset[0][train_index]
        trainset_y = dataset[1][train_index]
        testset_x = dataset[0][test_index]
        testset_y = dataset[1][test_index]
        trainset = list(zip(trainset_y,trainset_x))
        testset  = list(zip(testset_y,testset_x))
        p = clsfr.test(trainset,testset)
        pre.append(p)
        j += 1
        print(str(j)+'th fold:'+str(p) )
    return np.mean(pre)

if __name__ == '__main__':
    ham_path = '/Users/yty/Google Drive/CSHW/ML/enron1/ham/*.txt'
    spam_path = '/Users/yty/Google Drive/CSHW/ML/enron1/spam/*.txt'
    ham = read_file(ham_path)
    spam = read_file(spam_path)
    max_feature = 1000
    data_vec = embed(ham+spam,max_feature)
    dataset = [np.asarray(data_vec),np.concatenate((np.ones(len(ham)),np.zeros(len(spam))))]
    codata = list(zip(dataset[1],dataset[0]))
    shuffle(codata)