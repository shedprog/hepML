from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import os

from MlClasses.PerformanceTests import classificationReport,rocCurve,compareTrainTest,learningCurve
from MlClasses.Config import Config

#For cross validation and HP tuning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from asimovErrors import Z,eZ,asimov_scorer_function
from pandasPlotting.Plotter import Plotter

# Libs added at 17.07.18
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# class XGBClassifier(xgboost.XGBClassifier):
    
#     # take probabilities of the first class only
#     def predict_proba(self,X_test):
#         return super(XGBClassifier, self).predict_proba(X_test)[:,1]

#     # # Convert predict proba output to decision function
#     # # predict_proba = 1 / (1 + exp( -decision_function  ))
#     # def decision_function(self, X_test):

#     #     # if LEARN set is very bed, the error can appear
#     #     # ValueError: Input contains NaN, infinity or a value too large for dtype('float32')
#     #     return -np.log( 1.0 / self.class1_predict_proba - 1.0 )

#     # # definding score method
#     # def score(self, X_test, y_test):
#     #     return accuracy_score(y_test, self.predict(X_test))

class Bdt(object):
    '''Take some data split into test and train sets and train a bdt on it'''
    def __init__(self,data,output=None):

        self.data = data 
        self.output = output
        self.config=Config(output=output)

        self.score=None
        self.crossValResults=None

        # should be checked it valid for Bdt
        self.scoreTypes = ['acc']

        self.bestHyperOpr_score = 0
        self.bestHyperOpt_param = None

        self.scorer = metrics.make_scorer(asimov_scorer_function, 
            greater_is_better=True, needs_proba=False)

    # def setup(self,dtArgs={},bdtArgs={}):
    def setup(self,cls=XGBClassifier,**init_param):

        # We define cls and init_parameters to re-initialize them later
        # for the every iteration at hyper-parametr tunning
        self.cls = cls
        self.init_param = init_param

        #define our initial bdt
        self.bdt = self.cls(**self.init_param)

        self.config.addToConfig('Class used: ',self.bdt)
        self.config.addToConfig('Vars used: ',self.data.X.columns.values)
        self.config.addToConfig('BDT was activated: ',type(self.bdt).__name__)
        self.config.addToConfig('nEvalEvents',len(self.data.y_eval.index))
        self.config.addToConfig('nDevEvents',len(self.data.y_dev.index))
        self.config.addToConfig('nTrainEvents',len(self.data.y_train.index))
        self.config.addToConfig('nTestEvents',len(self.data.y_test.index))
        # self.config.addToConfig('DT arguments',dtArgs)
        # self.config.addToConfig('BDT arguments',bdtArgs)

    def fit(self):

        self.history = self.bdt.fit(self.data.X_train, self.data.y_train)

    def predict_proba(self,X_test):

        return self.bdt.predict_proba(X_test)[:,1]

    def crossValidation(self,kfolds=3,n_jobs=4):
        '''K-means cross validation'''
        self.crossValResults = cross_val_score(self.bdt, self.data.X_dev, self.data.y_dev,scoring='accuracy',n_jobs=n_jobs,cv=kfolds)
        self.config.addLine('CrossValidation')
        self.config.addToConfig('kfolds',kfolds)
        self.config.addLine('')
        
    def hyperopt_train_test_xgb(self,param):
        #function calculate accuracy and returns formated output for HyperOpt
        kfolds=3
        n_jobs=6
 
        acc = cross_val_score(self.cls(**dict(self.init_param,**param)), self.data.X_train, self.data.y_train,scoring=self.scorer,n_jobs=n_jobs,cv=kfolds).mean()

        if acc > self.bestHyperOpr_score:
            self.bestHyperOpr_score = acc
            self.bestHyperOpt_param = param

        print 'best:',self.bestHyperOpr_score, 'accuracy:', acc, param
        return {'loss': -acc, 'status': STATUS_OK}

    def print_best_hyperopt_results(self):
        print '~~~~~~~~~~~ best values ~~~~~~~~~~~~~~'
        print 'best:',self.bestHyperOpr_score
        print 'parametrs:',self.bestHyperOpt_param

    def gridSearch(self,param_grid,kfolds=3,n_jobs=4):
        '''Implementation of the sklearn grid search for hyper parameter tuning, 
        making use of kfolds cross validation.
        Pass a dictionary of lists of parameters to test on. Choose number of cores
        to run on with n_jobs, -1 is all of them'''

        grid = GridSearchCV(estimator=self.bdt,param_grid=param_grid,scoring='accuracy',n_jobs=n_jobs,cv=kfolds)
        self.gridResult = grid.fit(self.data.X_dev, self.data.y_dev)

        #Save the results
        if not os.path.exists(self.output): os.makedirs(self.output)
        outFile = open(os.path.join(self.output,'gridSearchResults.txt'),'w')
        outFile.write("Best: %f using %s \n\n" % (self.gridResult.best_score_, self.gridResult.best_params_))
        means = self.gridResult.cv_results_['mean_test_score']
        stds = self.gridResult.cv_results_['std_test_score']
        params = self.gridResult.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            outFile.write("%f (%f) with: %r\n" % (mean, stdev, param))
        outFile.close()

    def saveConfig(self):

        if not os.path.exists(self.output): os.makedirs(self.output)
        self.config.saveConfig()

    def classificationReport(self,doEvalSet=False):

        if doEvalSet: #produce report for dev and eval sets instead
            X_test=self.data.X_eval
            y_test=self.data.y_eval
            X_train=self.data.X_dev
            y_train=self.data.y_dev
            append='_eval'
        else:
            X_test=self.data.X_test
            y_test=self.data.y_test
            X_train=self.data.X_train
            y_train=self.data.y_train
            append=''

        if not os.path.exists(self.output): os.makedirs(self.output)
        f=open(os.path.join(self.output,'classificationReport'+append+'.txt'),'w')
        f.write( 'Performance on test set:')
        classificationReport(self.bdt.predict(X_test),self.predict_proba(X_test),y_test,f)

        f.write( '\n' )
        f.write('Performance on training set:')
        classificationReport(self.bdt.predict(X_train),self.predict_proba(X_train),y_train,f)

        if self.crossValResults is not None:
            f.write( '\n\nCross Validation\n')
            f.write("Cross val results: %.2f%% (%.2f%%)" % (self.crossValResults.mean()*100, self.crossValResults.std()*100))
        
    def rocCurve(self,doEvalSet=False):

        if doEvalSet: #produce report for dev and eval sets instead
            rocCurve(self.predict_proba(self.data.X_eval),self.data.y_eval,output=self.output,append='_eval')
            rocCurve(self.predict_proba(self.data.X_dev),self.data.y_dev,output=self.output,append='_dev')
        else:
            rocCurve(self.predict_proba(self.data.X_test),self.data.y_test,output=self.output)
            rocCurve(self.predict_proba(self.data.X_train),self.data.y_train,output=self.output,append='_train')

    def compareTrainTest(self,doEvalSet=False):
        if doEvalSet:
            compareTrainTest(self.predict_proba,self.data.X_dev,self.data.y_dev,\
                    self.data.X_eval,self.data.y_eval,self.output,append='_eval')
        else:
            compareTrainTest(self.predict_proba,self.data.X_train,self.data.y_train,\
                    self.data.X_test,self.data.y_test,self.output)

    def learningCurve(self,kfolds=3,n_jobs=1):
        learningCurve(self.bdt,self.data.X_dev,self.data.y_dev,self.output,cv=kfolds,n_jobs=n_jobs)

    def diagnostics(self,doEvalSet=False):

        self.saveConfig()
        self.classificationReport(doEvalSet=doEvalSet)
        self.rocCurve(doEvalSet=doEvalSet)
        self.compareTrainTest(doEvalSet=doEvalSet)

    def plotDiscriminator(self):
        plotDiscriminator(self.bdt,self.data.X_test,self.data.y_test, self.output)

    def testPrediction(self):
        return self.bdt.predict(self.data.X_test)

    def getAccuracy(self):
        if not self.score:
            self.score = self.bdt.score(self.data.X_test,self.data.y_test)
        return self.score

    def makeHepPlots(self,expectedSignal,expectedBackground,systematics=[0.0001],makeHistograms=True,subDir=None,customPrediction=None):
        '''Plots intended for binary signal/background classification
        
            - Plots significance as a function of discriminator output
            - Plots the variables for signal and background given different classifications 
            - If reference variables are available in the data they will also be plotted 
            (to be implemented, MlData.referenceVars)
        '''

        if subDir:
            oldOutput = self.output
            self.output=os.path.join(self.output,subDir)


        names = self.data.X.columns.values

        #Then predict the results and save them
        if customPrediction is None:
            # predictionsTest = self.testPrediction()
            predictionsTest = self.predict_proba(self.data.X_test)
        else:
            predictionsTest = customPrediction

        dataTest=pd.DataFrame(self.data.scaler.inverse_transform(self.data.X_test),columns=names)

        #Add the predictions and truth to a data frame
        dataTest['truth']=self.data.y_test.as_matrix()
        dataTest['pred']=predictionsTest

        signalSizeTest = len(dataTest[dataTest.truth==1])
        bkgdSizeTest = len(dataTest[dataTest.truth==0])
        signalWeightTest = float(expectedSignal)/signalSizeTest
        bkgdWeightTest = float(expectedBackground)/bkgdSizeTest

        def applyWeight(row,sw,bw):
            if row.truth==1: return sw
            else: return bw

        dataTest['weight'] = dataTest.apply(lambda row: applyWeight(row,signalWeightTest,bkgdWeightTest), axis=1)

        #save it for messing about
        #dataTest.to_pickle('dataTestSigLoss.pkl')

        #Produce a cumulative histogram of signal and background (truth) as a function of score
        #Plot it with a log scTrue

        h1=plt.hist(dataTest[dataTest.truth==0]['pred'],weights=dataTest[dataTest.truth==0]['weight'],bins=5000,color='b',alpha=0.8,label='background',cumulative=-1)
        h2=plt.hist(dataTest[dataTest.truth==1]['pred'],weights=dataTest[dataTest.truth==1]['weight'],bins=5000,color='r',alpha=0.8,label='signal',cumulative=-1)
        plt.yscale('log')
        plt.ylabel('Cumulative event counts / 0.02')
        plt.xlabel('Classifier output')
        plt.legend()
 
        plt.savefig(os.path.join(self.output,'cumulativeWeightedDiscriminator.pdf'))
        plt.clf()

        #From the cumulative histograms plot s/b and s/sqrt(s+b)

        s=h2[0]
        b=h1[0]

        plt.plot((h1[1][:-1]+h1[1][1:])/2,s/b)
        plt.title('sig/bkgd on test set')
        plt.savefig(os.path.join(self.output,'sigDivBkgdDiscriminator.pdf'))
        plt.clf()

        plt.plot((h1[1][:-1]+h1[1][1:])/2,s/np.sqrt(s+b))
        plt.title('sig/sqrt(sig+bkgd) on test set, best is '+str(max(s/np.sqrt(s+b))))
        plt.savefig(os.path.join(self.output,'sensitivityDiscriminator.pdf'))
        plt.clf()

        for systematic in systematics:
            # sigB=systematic*b
            #
            # toPlot=np.sqrt(2*( (s+b) * np.log( (s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB) ) - b*b*np.log( 1+sigB*sigB*s/(b*(b+sigB*sigB)) ) / (sigB*sigB) ))
            #plt.plot((h1[1][:-1]+h1[1][1:])/2,Z(s,b,systematic))
            toPlot = Z(s,b,systematic)
            plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot)
            es = signalWeightTest*np.sqrt(s/signalWeightTest)
            eb = bkgdWeightTest*np.sqrt(b/bkgdWeightTest)
            error=eZ(s,es,b,eb,systematic)
            # plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot-error)
            # plt.plot((h1[1][:-1]+h1[1][1:])/2,toPlot+error)
            plt.fill_between((h1[1][:-1]+h1[1][1:])/2,toPlot-error,toPlot+error,linewidth=0,alpha=0.6)
            maxIndex=np.argmax(toPlot)
            plt.title('Systematic '+str(systematic)+', s: '+str(round(s[maxIndex],1))+', b:'+str(round(b[maxIndex],1))+', best significance is '+str(round(toPlot[maxIndex],2))+' +/- '+str(round(error[maxIndex],2)))
            plt.xlabel('Cut on classifier score')
            plt.ylabel('Asimov estimate of significance')
            plt.savefig(os.path.join(self.output,'asimovDiscriminatorSyst'+str(systematic).replace('.','p')+'.pdf'))
            plt.clf()

            if makeHistograms: #Do this on the full set

                #Start with all the data and standardise it
                if self.data.standardised:
                    data = pd.DataFrame(self.data.scaler.transform(self.data.X))
                    #data = self.data.X.apply(self.data.scaler.transform)
                else:
                    data = self.data.X

                predictions= self.bdt.predict(data.as_matrix())
            
                #Now unstandardise
                if self.data.standardised:
                    data=pd.DataFrame(self.data.scaler.inverse_transform(data),columns=names)

                data['truth']=self.data.y.as_matrix()
                data['pred']=predictions

                signalSize = len(data[data.truth==1])
                bkgdSize = len(data[data.truth==0])
                signalWeight = float(expectedSignal)/signalSize
                bkgdWeight = float(expectedBackground)/bkgdSize

                data['weight'] = data.apply(lambda row: applyWeight(row,signalWeight,bkgdWeight), axis=1)

                #Plot all other interesting variables given classification
                p = Plotter(data,os.path.join(self.output,'allHistsSyst'+str(systematic).replace('.','p')))
                p1 = Plotter(data[data.pred>float(maxIndex)/len(toPlot)],os.path.join(self.output,'signalPredHistsSyst'+str(systematic).replace('.','p')))
                p2 = Plotter(data[data.pred<float(maxIndex)/len(toPlot)],os.path.join(self.output,'bkgdPredHistsSyst'+str(systematic).replace('.','p')))

                p.plotAllStackedHists1D('truth',weights='weight',log=True)
                p1.plotAllStackedHists1D('truth',weights='weight',log=True)
                p2.plotAllStackedHists1D('truth',weights='weight',log=True)

        if subDir:
            self.output=oldOutput
            
        pass