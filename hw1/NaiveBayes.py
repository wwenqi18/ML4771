#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMS W4771 Machine Learning
HW 1, Problem 6
Tianyang Yang, Wenqi Wang
October 5, 2018
"""

import numpy as np

""" Naive Bayes Classifier """
class NaiveBayes():
    def __init__(self):
        self.ham_prob, self.spam_prob = 0.5,0.5

    # compute ham and spam probs in the trainset 
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

    # compute the conditional probability of given test sample    
    def cond_prob(self,testsample):
        testcount = testsample[1]
        hamcount = [j for i,j in zip(testcount,self.ham_dict) if i!=0]
        spamcount = [k for i,k in zip(testcount,self.spam_dict) if i!=0]
        testcount = [i for i in testcount if i!=0]
        ham_cond = np.sum(testcount*(np.log((hamcount+np.ones(len(testcount)))/(self.ham_total+self.v))))+np.log(self.ham_prob)
        spam_cond = np.sum(testcount*(np.log((spamcount+np.ones(len(testcount)))/(self.spam_total+self.v))))+np.log(self.spam_prob)
        return ham_cond,spam_cond
    
    # perform classification for given trainset and testset
    def pred(self,trainset,testset):
        self.train(trainset)
        pred_label = []
        for testsample in testset:
            ham_cond, spam_cond = self.cond_prob(testsample)
            if ham_cond >= spam_cond:
                pred_label.append(1)
            else:
                pred_label.append(0)
        return pred_label
    
    # evaluate classification accuracy            
    def test(self,trainset,testset):
        pred_label = self.pred(trainset,testset)
        label = []
        for sample in testset:
            label.append(sample[0])
        precision = len([i for i,j in zip(pred_label,label) if i==j])/len(pred_label)
        return precision
