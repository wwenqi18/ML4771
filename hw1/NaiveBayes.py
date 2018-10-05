#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:23:14 2018

@author: yty
"""

import numpy as np

""" Naive Bayes Classifier """
class NaiveBayes():
    def __init__(self):
        self.ham_prob, self.spam_prob = 0.5,0.5

    def train(self,trainset):
        ham = [sample for sample in trainset if sample[0]==1]
        spam = [sample for sample in trainset if sample[0]==0]
        self.ham_prob = len(ham)/len(trainset)
        self.spam_prob = len(spam)/len(trainset)
        self.v = len(trainset[0][1])
        
        self.ham_dict = np.zeros(self.v)
        for trainsample in ham:
            self.ham_dict += trainsample[1]
        self.ham_total = np.sum(self.ham_dict)
        
        self.spam_dict = np.zeros(self.v)
        for trainsample in spam:
            self.spam_dict += trainsample[1]
        self.spam_total = np.sum(self.spam_dict)

    
        
    def cond_prob(self,testsample):
        testcount = testsample[1]
        hamcount = [j for i,j in zip(testcount,self.ham_dict) if i!=0]
        spamcount = [k for i,k in zip(testcount,self.spam_dict) if i!=0]
        testcount = [i for i in testcount if i!=0]
        ham_cond = np.sum(testcount*(np.log((hamcount+np.ones(len(testcount)))/(self.ham_total+self.v))))+np.log(self.ham_prob)
#        ham_cond = np.sum(testvec*(np.log(testvec+np.ones(len(testvec)))-np.log(np.sum(ham_matrix)+v)))+np.log(ham_prob)
        spam_cond = np.sum(testcount*(np.log((spamcount+np.ones(len(testcount)))/(self.spam_total+self.v))))+np.log(self.spam_prob)
#        spam_cond = np.sum(testvec*(np.log(testvec+np.ones(len(testvec)))-np.log(np.sum(spam_matrix)+v)))+np.log(spam_prob)
#        print('ham cond prob: '+str(ham_cond))
#        print('spam cond prob: ' + str(spam_cond))
        return ham_cond,spam_cond
        
    def pred(self,trainset,testset):
        self.train(trainset)
        pred_label = []
#        i = 0
        for testsample in testset:
#            print('test sample: '+str(i))
#            i += 1
            ham_cond, spam_cond = self.cond_prob(testsample)
            if ham_cond >= spam_cond:
                pred_label.append(1)
            else:
                pred_label.append(0)
        return pred_label
                
    def test(self,trainset,testset):
        pred_label = self.pred(trainset,testset)
        label = []
        for sample in testset:
            label.append(sample[0])
        precision = len([i for i,j in zip(pred_label,label) if i==j])/len(pred_label)
        return precision
