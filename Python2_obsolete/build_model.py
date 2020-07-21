"""Contains a functin that build quadratic model based on the sample.
"""

import numpy as np
import scipy

class quadraticModel:
    def __init__(self, n):
        self.H = np.empty((n,n))
        self.g = np.empty(n)

class linearModel:
    def __init__(self, n):
        self.g = np.empty(n)



def build_model(samp, par): 
    n, m = samp.n, samp.m - 1

    if (par.alg_model is 'quadratic') and (m > n):
        # fit a quadratic model 
        model = quadraticModel(n)

        # construct the big matrix
        A = np.empty((n+m, n+m))
        A[:n, :n] = np.zeros((n, n))
        A[:n, n:] = samp.Ycentered.T 
        A[n:, :n] = samp.Ycentered 
        A[n:, n:] = np.dot(samp.Ycentered, samp.Ycentered.T) ** 2 / 2

        # the right-hand side
        b = np.zeros(n+m) 
        b[n:] = samp.fcentered

        # solve the linear system and retrieve the gradient and the hessian 
        lamda = np.linalg.solve(A, b)
        model.g = lamda[:n]
        model.H = np.dot(samp.Ycentered.T * lamda[n:], samp.Ycentered)

    elif (par.alg_model is 'linear') and (m > n):
        # fit an over-determined linear model 
        model = linearModel(n)

        # linear regression
        model.g = np.linalg.lstsq(samp.Ycentered, samp.fcentered)[0]
        if np.isnan(model.g).any():
            # if np.linalg.lstsq goes crazy and returns nan
            A = samp.Ycentered
            model.g = np.linalg.solve(np.dot(A.T,A), np.dot(A.T, samp.fcentered))

    elif (par.alg_model is 'quadratic') and (m == n):
        # fit an well-determined linear model 
        model = quadraticModel(n)
        model.H = np.zeros((n,n))

        # solve the linear system
        model.g = np.linalg.solve(samp.Ycentered, samp.fcentered)

    elif (par.alg_model is 'linear') and (m == n):
        # fit an well-determined linear model 
        model = linearModel(n)

        # solve the linear system
        model.g = np.linalg.solve(samp.Ycentered, samp.fcentered)

    elif m < n:
        # fit an under-determined linear model 
        model = linearModel(n)

        # construct the small matrix
        A = np.dot(samp.Ycentered, samp.Ycentered.T)
        # ATTENTION! 
        # A here might be singular.
        # This issue is not addressed yet. 

        # solve the linear system and retrieve the gradient
        lamda = np.linalg.solve(A, samp.fcentered)
        model.g = np.dot(samp.Ycentered.T, lamda)


    
    return model


