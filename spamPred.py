#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 21:34:29 2018

@author: yty
"""

import numpy as np
from sklearn.model_selection import KFold
from random import shuffle
from collections import Counter




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



""" K-nearst Neighbors """
class KNN():
    def __init__(self,k,measure):
        self.k = k
        if measure not in ['1','2','inf']:
            raise ValueError('Unexpected metric type')
        self.metric = measure
        
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
        
        
    def test(self,trainset,testset):
        pred_label = self.pred(trainset,testset)
        label = []
        for sample in testset:
            label.append(sample[0])
        precision = len([i for i,j in zip(pred_label,label) if i ==j])/len(pred_label)
        return precision



""" Decision Tree Method """
class DTree():
    root = None
    
    def __init__(self):
        pass
    
    class Node():
        def __init__(self,feature,value):
            self.feature = feature
            self.value = value
            self.left = None
            self.right = None
            
        def add_children(self,left,right):
            self.left = left
            self.right = right
            return self
    
    class Leaf():
        def __init__(self,pred,prop):
            self.pred = pred
            self.prop = prop
    
    def major(self,dataset):
        labels = []
        for sample in dataset:
            labels.append(sample[0])
        if len(labels) == 0:
            return None,999
        pred = Counter(labels).most_common(1)[0][0]
        prop = pred/len(dataset)
        return pred,prop

    # compute entropy in a split consisted of split1 and split2
    def entropy(self,split1,split2):
#        ham1 = len([sample for sample in split1 if sample[0]==1])
#        ham2 = len([sample for sample in split2 if sample[0]==1])
#        prop1 = ham1/len(split1)
#        prop2 = ham2/len(split2)
#        entropy1 = prop1*np.log(1/prop1)+(1-prop1)*np.log(1/(1-prop1))
#        entropy2 = prop2*np.log(1/prop2)+(1-prop2)*np.log(1/(1-prop2))
#        
#        tot = len(split1)+len(split2)
#        tot_ham_prop = (ham1+ham2)/tot
#        tot_entropy = tot_ham_prop*np.log(1/tot_ham_prop)+(1-tot_ham_prop)*np.log(1/(1-tot_ham_prop))
#        return tot_entropy-prop1*entropy1-prop2*entropy2
        if len(split1) == 0 or len(split2) == 0:
            return 999
        pred1,prop1 = self.major(split1)
        prop1 = prop1/len(split1)
        pred2,prop2 = self.major(split2)
        prop2 = prop2/len(split1)
        entropy1 = prop1*np.log(1/prop1)+(1-prop1)*np.log(1/(1-prop1))
        entropy2 = prop2*np.log(1/prop2)+(1-prop2)*np.log(1/(1-prop2))
        
        tot = len(split1)+len(split2)
        tot_maj_prop = (prop1+prop2)/tot
        tot_entropy = tot_maj_prop*np.log(1/tot_maj_prop)+(1-tot_maj_prop)*np.log(1/(1-tot_maj_prop))
        return tot_entropy-prop1*entropy1-prop2*entropy2
    
    # return true/false if feature = yes/no for a sample
    def is_yes(self,sample,feature,value):
        if sample[1][feature] >= value:
            return True
        else:
            return False
    
    # return the no and yes groups if we split dataset with the given feature/value combo
    def split(self,dataset,feature,value):
        no,yes = [],[]
        for sample in dataset:
            if self.is_yes(sample,feature,value) == True:
                yes.append(sample)
            else:
                no.append(sample)
        print('no:'+str(len(no)))
        print('yes:'+str(len(yes)))
        return no,yes
    
    # return base cases if data is unambiguous or there are remaining features
    # else return the root of the decision tree with branches added    
    def train(self,trainset,remainf):
        threshold = 1-1e-5
        pred,prop = self.major(trainset)
        base = self.Leaf(None, None)
        opt_feature = 0
        opt_value = 0
        opt_entropy = 0
        if prop >= threshold or len(remainf) == 0:
            base.pred = pred
            base.prop = prop
            return base
        else:
            for f in remainf:
                for sample in trainset:
                    no,yes = self.split(trainset,f,sample[1][f])
                    print('f:'+str(f))
                    print('value:'+str(sample[1][f]))
                    cur_entropy = self.entropy(no,yes)
                    if cur_entropy > opt_entropy:
                        opt_entropy = cur_entropy
                        opt_feature = f
                        opt_value = sample[1][f]
            remainf.remove(opt_feature)
            left = self.train(no,remainf)
            right = self.train(yes,remainf)
            node = self.Node(opt_feature,opt_value)
            node.add_children(left,right)
            return node
    
    # return the predicted label of one test sample, given the decision tree
    def pred_sample(self,node,testsample):
        if isinstance(node,self.Leaf):
            return node.pred
        elif isinstance(node,self.Node):
            if self.is_yes(testsample,node.feature,node.value) == False:
                return self.pred_sample(node.left,testsample)
            else:
                return self.pred_sample(node.right,testsample)
    
    # train the decision tree with the given trainset and 
    # return predictions for the given testset
    def pred(self,trainset,testset):
        self.root = self.train(trainset,list(range(0,len(trainset[0][1]))))
        pred_label = []
        for testsample in testset:
            pred_label.append(self.pred_sample(self.root,testsample))
        return pred_label
        
    def test(self,trainset,testset):
        pred_label = self.pred(trainset,testset)
        label = []
        for sample in testset:
            label.append(sample[0])
        precision = len([i for i,j in zip(pred_label,label) if i ==j])/len(pred_label)
        return precision


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
#    knn = KNN(7,'2')
#    cv_knn = cv(knn,dataset,5)
#    nb = NaiveBayes()  
#    cv_nb = cv(nb,dataset,5)
    codata = list(zip(dataset[1],dataset[0]))
    shuffle(codata)
    dt = DTree()
    accuracy = dt.test(codata[0:200],codata[201:300])
    print(accuracy)