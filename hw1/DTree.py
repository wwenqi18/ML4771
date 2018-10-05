#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 01:20:46 2018

@author: yty
"""
import numpy as np
from collections import Counter


""" Decision Tree Method """
class DTree():
    
    def __init__(self,max_dep=20,thr = 1-1e-2):
        self.root = None
        self.depth = 1
        self.max_dep = max_dep
        self.threshold = thr
        
    
    class Node():
        def __init__(self,feature,value,dep=0):
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
        def __init__(self,pred,prop,dep=0):
            self.pred = pred
            self.prop = prop
            self.dep = dep
        def getDep(self):
            return self.dep
    
    def major(self,dataset):
#        if len(dataset) == 0:
#            return 1,1
        labels = []
        for sample in dataset:
            labels.append(sample[0])
        pred, count = Counter(labels).most_common(1)[0][0],Counter(labels).most_common(1)[0][1]
        prop = count/len(dataset)
        return pred,prop

    # compute entropy in a split consisted of split1 and split2
    def split_entropy(self,split1,split2):
        if len(split1) == 0 or len(split2) == 0:
            return -100
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
        
        tot = len(split1)+len(split2)
        return -(len(split1)/tot)*entropy1-(len(split1)/tot)*entropy2
    
    # return true/false if feature = yes/no for a sample
    def is_yes(self,sample,feature,value):
        if sample[1][feature] >= value:
            return True
        else:
            return False
    
    # return the no and yes groups if we split dataset with the given feature/value combo
    def split(self,dataset,feature,value):
        yes = [sample for sample in dataset if self.is_yes(sample,feature,value)]
        no = [sample for sample in dataset if not self.is_yes(sample,feature,value)]
        return no,yes
    
    def entropy(self,dataset):
        ham = len([sample for sample in dataset if sample[0]==1])
        spam = len([sample for sample in dataset if sample[0]==0])
        tot = len(dataset)
        if ham == 0 or spam == 0:
            return 0
        return -ham/tot*np.log2(ham/tot)-spam/tot*np.log2(spam/tot)
        
    # compute the informational gain of a feature in dataset
    def gain(self,dataset,feature):
        value_dict = {}
        for sample in dataset:
            value = sample[1][feature]
            if not value_dict.get(value):
                value_dict[value] = [1,sample[0],1-sample[0]]
            else:
                value_dict[value][0] += 1
                value_dict[value][2-int(sample[0])] += 1
        
        cond_ent = 0
        for value in value_dict:
            summary = value_dict.get(value)
            count,ham,spam = summary[0],summary[1],summary[2]
            if ham == 0 or spam == 0:
                pass
            else:
                cond_ent += count/len(dataset)*(-ham/count*np.log2(ham/count)-spam/count*np.log2(spam/count))
        if cond_ent == 0:
            return -100
        class_ent = self.entropy(dataset) 
        gain = class_ent - cond_ent
        return gain
    
    # return base cases if data is unambiguous or there are remaining features
    # else return the root of the decision tree with branches added    
    def train(self,trainset,remainf,cur_dep = 0):
        pred,prop = self.major(trainset)
        opt_feature = -1
        opt_gain = -100
        if prop >= self.threshold or len(remainf) == 0 or cur_dep >= self.max_dep :
            base = self.Leaf(None, None,dep = cur_dep)
            base.pred = pred
            base.prop = prop
            d = base.getDep()
            print('leaf depth: '+str(d))
            return base
        else:
            for f in remainf:
                info_gain = self.gain(trainset,f)
                if info_gain > opt_gain:
                    opt_feature = f
                    opt_gain = info_gain 
                if opt_gain == -100:
                    print('useless feature: '+str(f))
                    remainf.remove(f)    
            if opt_feature == -1:
                base = self.Leaf(None, None, dep = cur_dep)
                base.pred = pred
                base.prop = prop
                d = base.getDep()
                print('leaf depth: '+str(d))
                return base
            else:
#                print(self.gain(trainset,opt_feature))
                if self.gain(trainset,opt_feature) < 0.04:
                    base = self.Leaf(None, None, dep = cur_dep)
                    base.pred = pred
                    base.prop = prop
                    d = base.getDep()
                    print('leaf depth: '+str(d))
                    return base
                split_opt = -1000
                for sample in trainset:
                    no,yes = self.split(trainset,opt_feature,sample[1][opt_feature])
                    cur_entropy = self.split_entropy(no,yes)
#                    print(str(cur_entropy)+'\t'+str(split_opt)+'\t'+str(opt_feature))
                    if cur_entropy > split_opt:
                        split_opt = cur_entropy
                        opt_value = sample[1][opt_feature]
                remainf.remove(opt_feature)
                cur_remainf = remainf
                no,yes = self.split(trainset,opt_feature,opt_value)
                if len(no) == 0 or len(yes) == 0 :
                    base = self.Leaf(None, None,dep = cur_dep)
                    base.pred = pred
                    base.prop = prop
                    d = base.getDep()
                    print('leaf depth: '+str(d))
                    return base
                else:
                    left = self.train(no,cur_remainf,cur_dep+1)
                    right = self.train(yes,cur_remainf,cur_dep+1)
                    node = self.Node(opt_feature,opt_value,dep = cur_dep)
                    node.add_children(left,right)
                    print('successful partition\t feature: '+str(opt_feature))
                    return node
    
    # return the predicted label of one test sample, given the decision tree
    def pred_sample(self,node,testsample):
        if isinstance(node,self.Leaf):
            return node.pred
        elif isinstance(node,self.Node):
            if self.is_yes(testsample,node.feature,node.value):
                return self.pred_sample(node.right,testsample)
            else:
                return self.pred_sample(node.left,testsample)
    
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
    dt = DTree(60)
    accuracy = dt.test(codata[:4000],codata[4001:])
    print(accuracy)
#    from sklearn import tree
#    clf = tree.DecisionTreeClassifier()
#    clf.fit(dataset[0],dataset[1])
#    clf.predict(dataset[0][ind[3000:4000]])
