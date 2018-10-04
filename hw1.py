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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle



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
def embed(strings):
    vec = CountVectorizer(analyzer = 'word',max_features=100);
    X = vec.fit_transform(strings)
    X = X.toarray()
    print(vec.get_feature_names())
    #X = np.ndarray(shape = np.shape(X), buffer = X)
#    labels = [label] * X.shape[0]
#    ret = list(zip(labels, X))
    return X 
    
# main
if __name__ == '__main__':
    ham_path = '/Users/yty/Google Drive/CSHW/ML/enron1/ham/*.txt'
    spam_path = '/Users/yty/Google Drive/CSHW/ML/enron1/spam/*.txt'
    ham = read_file(ham_path)
    spam = read_file(spam_path)
    data_vec = embed(ham+spam)
    dataset = [np.asarray(data_vec),np.concatenate((np.ones(len(ham_vec)),np.zeros(len(spam_vec))))]
    codata = list(zip(dataset[1],dataset[0]))
    shuffle(codata)
    
    




