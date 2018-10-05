"""
COMS W4771 Machine Learning
HW 1, Problem 6
Tianyang Yang, Wenqi Wang
October 5, 2018

References:
https://machinelearningmastery.com/clean-text-machine-learning-python/
https://en.wikipedia.org/wiki/ID3_algorithm

"""

# import modules
import glob
import nltk
import numpy as np
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle
from sklearn.model_selection import KFold

from KNN import KNN
from NaiveBayes import NaiveBayes as NB
from DTree import DTree as DT


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
#    print(vec.get_feature_names())
    return X 

# perform k fold cross validation using the given dataset and classifier
# return the average accuracy
def cv(clsfr,dataset,k):
    pre = []
    kf = KFold(n_splits=k,shuffle=True)
    j = 0
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
        print('  '+str(j)+'th fold:'+str(p) )
    return np.mean(pre)

# run performance tests on the three classifiers
def performance_test(dataset):
    res = {'nb':[0,0,0,0],'knn':[0,0,0,0],'dt':[0,0,0,0]}
    nb = NB()
    knn = KNN(7,'2')
    dtree = DT(60)
    
    k_values = [2,3,5,10]
    i = 0
    for k in k_values:
        print('k={}'.format(k))
        print(' begin cv for nb')
        cv_nb = cv(nb,dataset,k)
        print(' nb:{}'.format(cv_nb))
        print(' begin cv for knn')
        cv_knn = cv(knn,dataset,k)
        print(' knn:{}'.format(cv_knn))
        print(' begin cv for dt')
        cv_dt = cv(dtree,dataset,k)
        print(' dt:{}'.format(cv_dt))
        
        res.get('nb')[i] = round(cv_nb,4)
        res.get('knn')[i] = round(cv_knn,4)
        res.get('dt')[i] = round(cv_dt,4)
        i += 1
    return res

def print_dict(dict,f):
    print('k=\t2\t3\t5\t10',file=f)
    for key,value in dict.items():
        print('{}\t{}\t{}\t{}\t{}'.format(key,value[0],value[1],value[2],value[3]),file=f)
    
# main
if __name__ == '__main__':
#    ham_path = '/Users/yty/Google Drive/CSHW/ML/enron1/ham/*.txt'
#    spam_path = '/Users/yty/Google Drive/CSHW/ML/enron1/spam/*.txt'
    ham_path = '/Users/wenqi/Desktop/enron1/ham/*.txt'
    spam_path = '/Users/wenqi/Desktop/enron1/spam/*.txt'
    print('loading dataset...')
    ham = read_file(ham_path)
    spam = read_file(spam_path)
    max_feature = 1000
    data_vec = embed(ham+spam,max_feature)
    dataset = [np.asarray(data_vec),np.concatenate((np.ones(len(ham)),np.zeros(len(spam))))]
    print('dataset loaded successfully')
#    codata = list(zip(dataset[1],dataset[0]))
#    shuffle(codata)
    
#    knn = KNN(7,'2')
#    cv_knn = cv(knn,dataset,5)
#    dtree = DT(60)
#    cv_dt = cv(dtree,dataset,3)
#    nb = NB()
#    cv_nb = cv(nb,dataset,5)
    res = performance_test(dataset)
    with open('output.txt','wt') as f:
        print_dict(res,f)
    
    
    
    
    




