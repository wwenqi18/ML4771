"""
COMS W4771 Machine Learning
HW 1, Problem 6
Tianyang Yang, Wenqi Wang
October 5, 2018

References:
https://machinelearningmastery.com/clean-text-machine-learning-python/
"""

import glob
import nltk
nltk.download('punkt')
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle
from sklearn.model_selection import KFold



# convert text files to list of strings
def read_file(path):
    strings = [];
    files = glob.glob(path)
    for file in files:
        f = open(file, 'r', errors='ignore')
        lines = f.read()
        f.close()
        tokens = word_tokenize(lines)
        words = [word for word in tokens if word.isalpha()]
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        stemmed = ' '.join(stemmed)
        strings.append(stemmed)     
    return strings

# convert list of strings to list vectors with labels
def embed(strings,max_f):
    vec = CountVectorizer(analyzer = 'word',max_features=max_f);
    X = vec.fit_transform(strings)
    X = X.toarray()
    print(vec.get_feature_names())
    #X = np.ndarray(shape = np.shape(X), buffer = X)
#    labels = [label] * X.shape[0]
#    ret = list(zip(labels, X))
    return X 


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

# main
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
    
    
    
    




