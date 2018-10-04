#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 01:20:46 2018

@author: yty
"""
import numpy as np
from sklearn.model_selection import KFold
from random import shuffle
from collections import Counter


""" Decision Tree Method """
class DTree():
    root = None
    
    def __init__(self):
        self.depth = 0
    
    class Node():
        def __init__(self,feature,value,dep):
            self.feature = feature
            self.value = value
            self.left = None
            self.right = None
            self.dep = dep
            
        def add_children(self,left,right):
            self.left = left
            self.right = right
            return self
        def getDep(self):
            return self.dep
    
    class Leaf():
        def __init__(self,pred,prop,dep):
            self.pred = pred
            self.prop = prop
            self.dep = dep
        def getDep(self):
            return self.dep
    
    def major(self,dataset):
        if len(dataset) == 0:
            return 1,1
        labels = []
        for sample in dataset:
            labels.append(sample[0])
        pred, count = Counter(labels).most_common(1)[0][0],Counter(labels).most_common(1)[0][1]
        prop = count/len(dataset)
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
            return -999
        pred1,prop1 = self.major(split1)
        pred2,prop2 = self.major(split2)
        if prop1 == 1:
            entropy1 = 0
        else:
            entropy1 = prop1*np.log(1/prop1)+(1-prop1)*np.log(1/(1-prop1)) 
        if prop2 == 1:
            entropy2 = 0
        else:
            entropy2 = prop2*np.log(1/prop2)+(1-prop2)*np.log(1/(1-prop2))
        
#        tot = len(split1)+len(split2)
#        tot_maj_prop = (prop1+prop2)/tot
#        tot_entropy = tot_maj_prop*np.log(1/tot_maj_prop)+(1-tot_maj_prop)*np.log(1/(1-tot_maj_prop))
        return -(len(split1)/len(split1)+len(split2))*entropy1-(len(split1)/len(split1)+len(split2))*entropy2
    
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
#        print('no:'+str(len(no)))
#        print('yes:'+str(len(yes)))
        return no,yes
    
    # return base cases if data is unambiguous or there are remaining features
    # else return the root of the decision tree with branches added    
    def train(self,trainset,remainf):
        threshold = 1-1e-1
        pred,prop = self.major(trainset)
        opt_feature = 0
        opt_value = 0
        opt_entropy = -100
        if prop >= threshold or len(remainf) == 0:
            base = self.Leaf(None, None)
            base.pred = pred
            base.prop = prop
            return base
        else:
            for f in remainf:
                for sample in trainset:
                    no,yes = self.split(trainset,f,sample[1][f])
#                    print('f:'+str(f))
#                    print('value:'+str(sample[1][f]))
                    cur_entropy = self.entropy(no,yes)
                    print(cur_entropy)
                    if cur_entropy > opt_entropy:
                        opt_entropy = cur_entropy
                        opt_feature = f
                        opt_value = sample[1][f]
                        print('successful partition')
            remainf.remove(opt_feature)
            no,yes = self.split(trainset,opt_feature,opt_value)
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


if __name__ == '__main__':
    dt = DTree()
    accuracy = dt.test(codata[:400],codata[401:500])
    print(accuracy)
#    from sklearn import tree
#    clf = tree.DecisionTreeClassifier()
#    clf.fit(dataset[0],dataset[1])
#    clf.predict(dataset[0][ind[3000:4000]])
