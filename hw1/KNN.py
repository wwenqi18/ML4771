#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMS W4771 Machine Learning
HW 1, Problem 6
Tianyang Yang, Wenqi Wang
October 5, 2018
"""


import numpy as np


""" K-nearst Neighbors """
class KNN():
    def __init__(self,k,measure):
        self.k = k
        if measure not in ['1','2','inf']:
            raise ValueError('Unexpected metric type')
        self.metric = measure
    
    # perform classification for given trainset and testset    
    def pred(self,trainset,testset):
        pred_label = []   
        for testsample in testset:
            knearest = [[],[]]
            for trainsample in trainset:
                if self.metric == '2':
                    d = np.sum((trainsample[1]-testsample[1])**2)
                elif self.metric == '1':
                    d = np.sum(np.abs(trainsample[1]-testsample[1]))
                elif self.metric == 'inf':
                    d = np.max(np.abs(trainsample[1]-testsample[1]))
                if len(knearest[0])< self.k:
                    knearest[0].append(d)
                    knearest[1].append(trainsample[0])
                elif d < max(knearest[0]):
                    i = knearest[0].index(max(knearest[0]))
                    knearest[0][i] = d
                    knearest[1][i] = trainsample[0]
            if 2*sum(knearest[1]) >= self.k:
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
        precision = len([i for i,j in zip(pred_label,label) if i ==j])/len(pred_label)
        return precision
