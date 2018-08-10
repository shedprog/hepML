from numpy import log,power,sqrt
from keras import backend as K
eps=0.000001

# s,b cnts observed in experimentm 
# sig relative error on b

# Asimov significance
def Z(s,b,sig=None):
    #if sig == None: sig=eps
    return sqrt( -2.0/(sig*sig)*log( b/( eps + b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)+ 2.0*( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/(eps + (b*b)+( b+s)*(b*b)*(sig*sig))+eps))

# error propagation on Asimov significance
def eZ(s,es,b,eb,sig=None):
    #if sig == None: sig=eps
    #if sig < eps: sig=eps # to avoid stability issue in calculation
    return power(-(eb*eb)/( 1.0/(sig*sig)*log( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)-( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))))*power( 1.0/( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)/(sig*sig)*( 1.0/( b+(b*b)*(sig*sig))*(sig*sig)*s-b/power( b+(b*b)*(sig*sig),2.0)*(sig*sig)*( 2.0*b*(sig*sig)+1.0)*s)-( ( b+s)*( 2.0*b*(sig*sig)+1.0)/( (b*b)+( b+s)*(b*b)*(sig*sig))+( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))-( b+s)*( 2.0*( b+s)*b*(sig*sig)+2.0*b+(b*b)*(sig*sig))*( b+(b*b)*(sig*sig))/power( (b*b)+( b+s)*(b*b)*(sig*sig),2.0))/( b+(b*b)*(sig*sig))*( (b*b)+( b+s)*(b*b)*(sig*sig))-log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))),2.0)/2.0-1.0/( 1.0/(sig*sig)*log( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)-( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))))*power( log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig)))+1.0/( b+(b*b)*(sig*sig))*( ( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))-( b+s)*(b*b)*( b+(b*b)*(sig*sig))*(sig*sig)/power( (b*b)+( b+s)*(b*b)*(sig*sig),2.0))*( (b*b)+( b+s)*(b*b)*(sig*sig))-1.0/( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)*b/( b+(b*b)*(sig*sig)),2.0)*(es*es)/2.0,(1.0/2.0))    

# n_s,n_b number of events used for testing etc.
def wghtd_Z(scale_s,n_s,scale_b,n_b,sig=None):
    return Z(scale_s*n_s,scale_b*n_b,sig)

def wghtd_eZ(scale_s,n_s,scale_b,n_b,sig=None):
    return eZ(scale_s*n_s,scale_s*sqrt(n_s),scale_b*n_b,scale_b*sqrt(n_b),sig)

def asimov_scorer_function(y_true,y_pred,sig = 0.001):
	# global expectedSignal,expectedBkgd
	# _all = y_true[y_pred == 1]
	# signal, back = len(_all[_all == 1]), len(_all[_all == 0])
	lumi=30. #luminosity in /fb
	expectedBkgd=844000.*8.2e-4*lumi #cross section of ttbar sample in fb times efficiency measured by Marco
	expectedSignal=228.195*0.14*lumi 

	signalWeight=expectedSignal/sum(y_true)# /len(y_true[y_true ==])
	bkgdWeight=expectedBkgd/sum(1-y_true)

	s = signalWeight*sum(y_pred*y_true)
	b = bkgdWeight*sum(y_pred*(1-y_true))
	print "signal = ",s,"bkgd = ",b
	return 1./Z(s, b, sig=sig)

def asimov_eval_matrics(y_pred, dataMATRIX):
  	y_true = dataMATRIX.get_label()
	return 'asimov_loss',float(asimov_scorer_function(y_true,y_pred,sig = 0.001))

# example usage
# Z(14,5,0.3)                   : 3.6149635359712184
# eZ(14,sqrt(14),5,sqrt(5),0.3) : 1.053697793635855
# wghtd_Z(2,7,10,0.5,0.3)       : 3.6149635359712184
# wghtd_eZ(2,7,10,0.5,0.3)      : 2.605193580282292

