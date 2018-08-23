import numpy as np
from  pandas import *
import math
from scipy import optimize
from numpy import log,power,sqrt

def asimov(X, sig):
    #if sig == None: sig=eps
    eps=0.000001
    s,b = X[0],X[1]

    result = 1./(sqrt( -2.0/(sig*sig)*log( b/( eps + b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)+\
        2.0*( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/(eps + (b*b)+( b+s)*(b*b)*(sig*sig))+eps)))
    return result

def asimov_obj(y_true,preds,expectedSignal=None,expectedBkgd=None,sigma=0.1):

    # print y_true,preds
    y = y_true
    # weight = dtrain.get_weight()

    yhat = 1.0 / (1.0 + np.exp(-preds))
    
    
    #print 'expected:',expectedSignal ,expectedBkgd
    signalWeight=float(expectedSignal)/np.sum(y_true)
    bkgdWeight=float(expectedBkgd)/np.sum(1-y_true) 

    #print 'weights:', signalWeight, bkgdWeight
    # approx ams using soft probability instead of hard predicted class
    #s = np.sum( weight * y * yhat )
    #b = np.sum( weight * (1.-y) * yhat )
    s = np.sum( y * yhat ) * signalWeight 
    b = np.sum( (1.-y) * yhat ) * bkgdWeight 
    #print "signal:",s," back:", b
    eps = np.sqrt(np.finfo(float).eps)
    s_prime,b_prime = optimize.approx_fprime([s,b], asimov, [eps*s,eps*b],sigma) 

    # grad = (s_prime * y + b_prime * (1.-y)) * weight * yhat * (1.-yhat)
    grad = -(s_prime*(yhat-y)+b_prime*(1-(yhat-y)))
    # grad = -(s_prime*y+b_prime*(1-y))
    # grad = optimize.approx_fprime(yhat, asimov, eps*yhat, y, weight) 
    hess = np.ones(yhat.shape)/(10**3) #constant

    #print grad, hess
    return grad, hess
