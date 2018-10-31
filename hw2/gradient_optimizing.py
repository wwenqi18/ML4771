#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 00:41:16 2018

@author: Tianyang Yang, Wenqi Wang
"""

import numpy as np
from scipy import optimize

def subset(full,prop=0.1):
    N = full.shape[0]
    index = np.random.choice(N,int(prop*N),replace = False)
    return full[index]

def f(theta,sample):
    """
    compute the LSE loss based on theta and smaple
    return a real number
    """
    m = sample.shape[0]
    pred = np.matmul(sample[:,1:],theta)
    pred = np.matrix(pred).T
    respond = sample[:,0]
    return np.linalg.norm(respond-pred)/m


def grad(f,theta,sample):
    """
    @f: a loss function on theta (in R^d) to R
    @theta: a d-dimension nd-array
    @sample: the dataset used to compute the loss (m x (d+1))
    return a d-dimension array grad
    """
    delta = 1e-7
    divdelta = 1e7
    d = theta.size
    grad = np.zeros(d)
    for i in range(d):
        t1, t2 = np.zeros(d), np.zeros(d)
        t1[i], t2[i] = -0.5*delta, 0.5*delta
        t1 = t1 + theta
        t2 = t2 + theta
        grad[i] = (f(t2,sample)-f(t1,sample))*divdelta
    return grad


def gd(f,theta,full,alpha=0.1,epsilon=1e-4,K=1e4):
    """
    @f: a loss function on theta (in R^d) to R
    @theta: a d-dimension nd-array, the initial value
    @full: the dataset used to compute the loss
    @alpha: step size
    @epsilon: rule of stop
    @K: the maximum number of iteration
    return the result and the iteration times
    """
    for i in range(int(K)):
        step = -alpha*grad(f,theta,full)
        if np.linalg.norm(step,np.inf) < epsilon:
            return theta,i
        else:
            theta = theta + step
    return theta,K


def sgd(f,theta,full,alpha=0.1,epsilon=1e-4,K=1e4):
    """
    @f: a loss function on theta (in R^d) to R, only take a random subset 
    @theta: a d-dimension nd-array, the initial value
    @full: the dataset used to draw random subset
    @alpha: step size
    @epsilon: rule of stop
    @K: the maximum number of iteration
    return the result and the iteration times
    """
    for i in range(int(K)):
        step = -alpha*grad(f,theta,subset(full))
        if np.linalg.norm(step,np.inf) < epsilon:
            return theta,i
        else:
            theta = theta + step
    return theta,K



def sgdm(f,theta,full,alpha=0.1,epsilon=1e-4,K=1e4,eta=0.9):
    """
    @f: a loss function on theta (in R^d) to R, only take a random subset 
    @theta: a d-dimension nd-array, the initial value
    @sample: the dataset used to draw random subset
    @alpha: step size
    @epsilon: rule of stop
    @K: the maximum number of iteration
    @eta: the decay rate of old gradient (momentum parameter)
    return the result and the iteration times
    """
    v = np.zeros(theta.shape[0])
    for i in range(int(K)):
        v = eta*v-grad(f,theta+eta*v,subset(full))
        step = alpha*v
        if np.linalg.norm(step,np.inf) < epsilon:
            return theta,i
        else:
            theta = theta + step
    return theta,K
    
    
    
def adagrad(f,theta,full,alpha=0.1,epsilon=1e-4,K=1e4):
    """
    @f: a loss function on theta (in R^d) to R, only take a random subset 
    @theta: a d-dimension nd-array, the initial value
    @full: the full dataset used to draw random subset
    @alpha: step size
    @epsilon: rule of stop
    @K: the maximum number of iteration
    return the result and the iteration times
    """
    stable = 1e-7
    d = theta.shape[0]
    r = np.zeros(d)
    for i in range(int(K)):
        g = grad(f,theta,subset(full))
        r = r + g*g
        step = -alpha/(np.repeat(stable,d)+np.sqrt(r))*g
        if np.linalg.norm(step,np.inf) < epsilon:
            return theta,i
        else:
            theta = theta + step
    return theta, K


def rmsprop(f,theta,full,alpha=0.1,epsilon=1e-4,K=1e4,rho=0.9):
    """
    @f: a loss function on theta (in R^d) to R, only take a random subset 
    @theta: a d-dimension nd-array, the initial value
    @full: the full dataset used to draw random subset
    @alpha: step size
    @epsilon: rule of stop
    @K: the maximum number of iteration
    @rho: decay rate of squared gradient
    return the result and the iteration times
    """
    stable = 1e-7
    d = theta.shape[0]
    r = np.zeros(d)
    for i in range(int(K)):
        g = grad(f,theta,subset(full))
        r = rho*r + (1-rho)*g*g
        step = -alpha/(np.sqrt(np.repeat(stable,d)+r))*g
        if np.linalg.norm(step,np.inf) < epsilon:
            return theta,i
        else:
            theta = theta + step
    return theta, K


    
def adadelta(f,theta,full,alpha=0.1,epsilon=1e-4,K=1e4,rho=0.9,stable=1e-7):
    """
    @f: a loss function on theta (in R^d) to R, only take a random subset 
    @theta: a d-dimension nd-array, the initial value
    @full: the full dataset used to draw random subset
    @alpha: step size
    @epsilon: rule of stop
    @K: the maximum number of iteration
    @rho: decay rate of squared gradient
    return the result and the iteration times
    """
    d = theta.shape[0]
    s, r = np.zeros(d), np.zeros(d)
    for i in range(int(K)):
        g = grad(f,theta,subset(full))
        r = rho*r + (1-rho)*g*g
        step = -np.sqrt(s+np.repeat(stable,d))/np.sqrt(r+np.repeat(stable,d))*g
        s = rho*s + (1-rho)*(step**2)
        if np.linalg.norm(step,np.inf) < epsilon:
            return theta,i
        else:
            theta = theta + step
    return theta, K 
    
    
def adam(f,theta,full,alpha=0.1,epsilon=1e-4,K=1e4,rho1=0.9,rho2=0.999,stable=1e-8):
    """
    @f: a loss function on theta (in R^d) to R, only take a random subset 
    @theta: a d-dimension nd-array, the initial value
    @full: the full dataset used to draw random subset
    @alpha: step size
    @epsilon: rule of stop
    @K: the maximum number of iteration
    @rho1, rho2: decay rate of moments
    @stable: constant for numerical stability
    return the result and the iteration times
    """
    d = theta.shape[0]
    s, r = np.zeros(d), np.zeros(d)
    for i in range(int(K)):
        g = grad(f,theta,subset(full))
        s = rho1*s + (1-rho1)*g
        r = rho2*r + (1-rho2)*g*g
        shat = s/(1-rho1**(i+1))
        rhat = r/(1-rho2**(i+1))
        step = -alpha*shat/(np.sqrt(rhat)+np.repeat(stable,d))
        if np.linalg.norm(step,np.inf) < epsilon:
            return theta,i
        else:
            theta = theta + step
    return theta, K

if __name__ == '__main__':
    d = 10
    N = 1000
    X1 = np.random.randn(N,d)
    theta1 = np.random.poisson(1,d)
    Y1 = np.matmul(X1,theta1)+np.random.randn(N)
    Y1 = np.matrix(Y1).T
    full = np.concatenate((Y1,X1),1)
    