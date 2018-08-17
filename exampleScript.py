import os
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use('Agg') #this stops matplotlib trying to use Xwindows backend when running remotely
import matplotlib.pyplot as plt
import pandas as pd

from keras import callbacks

from MlClasses.MlData import MlData
from MlClasses.Dnn import Dnn

# from MlClasses.Bdt import Bdt
from MlClasses.Bdt import *

#===== Define some useful variables =====
#
# makePlots=False
# doClassification=False
# doRegression=False

parser = argparse.ArgumentParser()
parser.add_argument("-plot","--makePlots", help = "Make a couple of plots",
                    action="store_true")
parser.add_argument("-class","--doClassification", help = "Make a simple network to carry out classification",
                    action="store_true")
parser.add_argument("-reg","--doRegression", help = "Make a simple network to carry out regressions",
                    action="store_true")
parser.add_argument("-bdt","--doBdtClassification", help = "Make a simple network to carry out classification by sklearn.tree",
                    action="store_true")
parser.add_argument("-xgb","--doXGBClassification", help = "Make a simple network to carry out classification by XGBoost",
                    action="store_true")
parser.add_argument("-hp","--doHyperOpt", help = "Make hyper-parametr optimization for XGBoost classification",
                    action="store_true")
args = parser.parse_args()

makePlots=args.makePlots
doClassification=args.doClassification
doRegression=args.doRegression
doBdtClassification=args.doBdtClassification
doXGBClassification=args.doXGBClassification
doHyperOpt=args.doHyperOpt

print "makePlots           ==>", makePlots
print "doClassification    ==>", doClassification
print "doRegression        ==>", doRegression
print "doBdtClassification ==>", doBdtClassification
print "doXGBClassification ==>", doXGBClassification
print "doHyperOpt          ==>", doHyperOpt

output='exampleOut' # an output directory (then make it if it doesn't exist)
if not os.path.exists(output): os.makedirs(output)

#class to stop training of dnns early
earlyStopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2)

lumi=30. #luminosity in /fb
expectedBkgd=844000.*8.2e-4*lumi #cross section of ttbar sample in fb times efficiency measured by Marco

#ttbar background and stop (900,100) 
# df = pd.read_pickle('/nfs/dust/cms/user/elwoodad/dlNonCms/hepML/dfs/combined.pkl')
# expectedSignal=17.6*0.059*lumi #cross section of stop sample in fb times efficiency measured by Marco

#ttbar background and stop (600,400) 
# dfFull = pd.read_pickle('/nfs/dust/cms/user/elwoodad/dlNonCms/hepML/dfs/combinedleonid.pkl')
expectedSignal=228.195*0.14*lumi 

#read Higgs data
# initial_names = ['signal','lepton_pT','lepton_eta','lepton_phi','missing_energy_magnitude', 'missing_energy_phi',
# 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
# 'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
# 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
# dfFull = pd.read_csv('/nfs/dust/cms/user/elwoodad/higgsDataset/HIGGS2M_1.csv' , sep = ',',
# names = initial_names)


dfTrain = pd.read_csv('/nfs/dust/cms/user/mykytaua/Data/Higgs_kaggle/training.csv', sep = ',', header=0, 
        na_values=-999,index_col=0)

#print "Nan columns:", train[train.isnull().any(axis=1)].head() #['DER_mass_MMC'].head()

# Some cosmetic
size_all = dfTrain.shape[0]
dfTrain = dfTrain.dropna()
dfTrain['Label'] = dfTrain['Label'].replace({'s': 1, 'b': 0})

size_3jet_only =  dfTrain.shape[0]

re_scale = size_all/size_3jet_only
dfTrain['Weight'] = dfTrain['Weight'] * re_scale

#===== Load the data from a pickle file (choose one of the two below) =====

#Pick a subset of events to limit size for messing about
#be careful to pick randomly as the first half are signal and the second half background
# dfFull = dfFull.sample(500000,random_state=42)

#Look at the variables in the trees:

print 'The train set keys are:'
print dfTrain.keys()

#Define and select a subset of the variables:

# subset=['signal', #1 for signal and 0 for background
#         'HT','MET', #energy sums
#         'MT','MT2W', #topological variables
#         'n_jet','n_bjet', #jet and b-tag multiplicities
#         'sel_lep_pt0','sel_lep_eta0','sel_lep_phi0', #lepton 4-vector
#         'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',# lead jet 4-vector
#         'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',# second jet 4-vector
#         'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2']# third jet 4-vector
# df=dfFull[subset]

df = dfTrain
print 'back: ',len(df[df.Label==0]['Label'])
print 'signal: ',len(df[df.Label==1]['Label'])


# print 'The reduced keys are:'
# print df.keys()




if makePlots:
    #===== Make a couple of plots: =====

    #Calculate the weights for each event and add them to the dataframe
    # signalWeight = expectedSignal/(df.signal==1).sum() #divide expected events by number in dataframe
    # bkgdWeight   = expectedBkgd/(df.signal==0).sum()

    #Add a weights column with the correct weights for background and signal
    # df['weight'] = df['signal']*signalWeight+(1-df['signal'])*bkgdWeight

    #Choose some variables to plot and loop over them
    #varsToPlot = [u'DER_mass_MMC', u'DER_mass_transverse_met_lep', u'DER_mass_vis',
    #   u'DER_pt_h', u'DER_deltaeta_jet_jet', u'DER_mass_jet_jet',
    #   u'DER_prodeta_jet_jet', u'DER_deltar_tau_lep', u'DER_pt_tot',
    #   u'DER_sum_pt', u'DER_pt_ratio_lep_tau', u'DER_met_phi_centrality',
    #   u'DER_lep_eta_centrality', u'PRI_tau_pt', u'PRI_tau_eta',
    #   u'PRI_tau_phi', u'PRI_lep_pt', u'PRI_lep_eta', u'PRI_lep_phi',
    #   u'PRI_met', u'PRI_met_phi', u'PRI_met_sumet', u'PRI_jet_num',
    #   u'PRI_jet_leading_pt', u'PRI_jet_leading_eta', u'PRI_jet_leading_phi',
    #   u'PRI_jet_subleading_pt', u'PRI_jet_subleading_eta',
    #   u'PRI_jet_subleading_phi', u'PRI_jet_all_pt',]
    varsToPlot = [u'DER_mass_MMC']	
    
    for v in varsToPlot:

        print 'Plotting',v
        maxRange=max(df[v])
        minRange=min(df[v])
	#Plot the signal and background but stacked on top of each other
        plt.hist([df[df.Label==1][v],df[df.Label==0][v]], #Signal and background input
                label=['background','signal'],
                bins=50, range=[minRange,maxRange], 
                stacked=True, color = ['r','g'],
                weights= [df[df.Label==1]['Weight'],df[df.Label==0]['Weight']])
                 #weights=[df[df.signal==0]['weight'],df[df.signal==1]['weight']]) #supply the weights

        #plt.hist(df[df.Label==0][v], #Signal and background input
        #        label='background',
        #        bins=50, range=[minRange,maxRange], 
        #        stacked=True, color = 'g',alpha=1,
        #        weights= df[df.Label==0]['Weight'])
                #weights=[df[df.signal==0]['weight'],df[df.signal==1]['weight']]) #supply the weights
        
        #histogram signal
        #plt.hist(df[df.Label==1][v], #Signal and background input
        #        label='signal',
        #        bins=50, range=[minRange,maxRange], 
        #        stacked=True, color = 'r',alpha=0.8,
        #        weights= df[df.Label==1]['Weight'])
        #        #weights=[df[df.signal==0]['weight'],df[df.signal==1]['weight']]) #supply the weights

        #histogram background
        
        plt.yscale('log')
        plt.xlabel(v)
        plt.legend()
        plt.savefig(os.path.join(output,'hist_'+v+'.pdf')) #save the histogram
        plt.clf() #Clear it for the next one

    df = df.drop('Weight',axis=1) #drop the weight to stop inference from it as truth variable

if doClassification:

    #=============================================================
    #===== Make a simple network to carry out classification =====
    #=============================================================

    print 'Running classification'
    print '-----Timer start-----'
    start_time = time.time()

    # here I make use of the hepML framework with keras
    # aim is to correctly classify signal or background events

    #===== Prepare the data =====
    # use an MlData class to wrap and manipulate the data with easy functions

    print 'Preparing data'

    mlDataC = MlData(df.drop('Weight',axis=1),'Label',weights=df['Weight']) #insert the dataframe and tell it what the truth variable is
    mlDataC.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now

    #Now decide whether we want to standardise the dataset
    #it is worth seeing what happens to training with and without this option
    #(this must be done after the split to avoid information leakage)
    mlDataC.standardise()

    #===== Setup and run the network  =====

    print 'Setting up network'

    dnnC = Dnn(mlDataC,os.path.join(output,'classification')) #give it the data and an output directory for plots

    #build a 2 hidden layer model with 50 neurons each layer
    #Note: if the number of neurons is a float it treats it as a proportion of the input
    # loss is binary cross entropy and one sigmoid neuron is used for output
    dnnC.setup(hiddenLayers=[1.0],dropOut=None,l2Regularization=None,loss='binary_crossentropy')


    #fit the defined network with the data passed to it
    #define an early stopping if the loss stops decreasing after 2 epochs
    print 'Fitting'
    dnnC.fit(epochs=200,batch_size=128,callbacks=[earlyStopping])

    #now produce some diagnostics to see how it went
    print 'Making diagnostics'
    dnnC.diagnostics() #generic diagnostics, ROC curves etc
    # hep specific plots including sensitivity estimates with a flat systematic etc: 
    print '\nMaking HEP plots'
    #dnnC.makeHepPlots(expectedSignal,expectedBkgd,systematics=[0.2],makeHistograms=False)
    dnnC.makeHepPlots(systematics=[0.1],makeHistograms=False)

    print '----Timer stop----'
    print 'General CPU time: ', time.time()-start_time

if doBdtClassification:

    print 'Running BdtClassification'
    print '-----Timer start-----'
    start_time = time.time()

    print 'Preparing data'

    mlDataC = MlData(df,'signal') #insert the dataframe and tell it what the truth variable is
    mlDataC.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now 

    # print 'Data output before standardise:'
    # mlDataC.output(number_of_lines=5)

    mlDataC.standardise()

    print 'Data output after standardise:'
    mlDataC.output(number_of_lines=5)

    print 'Defining BDT'
    bdt = Bdt(mlDataC,output+'/DBT')

    print 'Setup BDT'
    bdt.setup(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,min_samples_leaf=0.05),
                          algorithm='SAMME',n_estimators=1000, learning_rate=0.5))
    # ====================DecisionTreeClassifier args=========================
    # min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
    # max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
    # class_weight=None, presort=False
    # ====================AdaBoostClassifier Args=============================
    # base_estimator=None, n_estimators=50, learning_rate=1.0, 
    # algorithm=SAMME.R, random_state=None  

    print 'Fitting BDT'
    bdt.fit()

    print 'Diagnostic BDT'  
    bdt.diagnostics()

    print 'Making HEP plots'
    bdt.makeHepPlots(expectedSignal,expectedBkgd,systematics=[0.2],makeHistograms=False)

    print '----Timer stop----'
    print 'General CPU time: ', time.time()-start_time


if doXGBClassification:

    print 'Running XGBClassification'
    print '------Timer start--------'
    start_time = time.time()

    print 'Preparing data'

    mlDataC = MlData(df,'signal') #insert the dataframe and tell it what the truth variable is
    mlDataC.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now 

    # print 'Data output before standardise:'
    # mlDataC.output(number_of_lines=5)

    mlDataC.standardise()

    print 'Data output after standardise:'
    mlDataC.output(number_of_lines=5)

    print 'Defining XGB'
    bdt = Bdt(mlDataC,output+'/XGB',separation_facet=0.957)

    print 'Setup XGB'
    
    bdt.setup(cls=XGBClassifier,max_depth=1,gamma=0.3,min_child_weight=18)
    # ========================XGBClassifier args===============================
    # max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, 
    # objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, 
    # gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,colsample_bytree=1,
    # colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
    # random_state=0, seed=None, missing=None, **kwarg

    print 'Fitting XGB'
    bdt.fit()

    print 'Diagnostic XGB'  
    bdt.diagnostics()

    print 'Making HEP plots'
    bdt.makeHepPlots(expectedSignal,expectedBkgd,systematics=[0.2],makeHistograms=False)

    print '----Timer stop----'
    print 'General CPU time: ', time.time()-start_time

if doRegression:

    #=========================================================
    #===== Make a simple network to carry out regression =====
    #=========================================================

    #now we've seen a classification example, try a similar thing with regression
    #try to predict a higher level variable from the low level inputs

    print 'Running regression'

    print 'Preparing data'

    #Just pick the 4-vectors to train on
    subset = ['HT']
    for k in dfFull.keys():
        for v in ['selJet','sel_lep']:
            if ' '+v in ' '+k: subset.append(k)

    print 'Using subset',subset
    df=dfFull[subset]

    df=df.fillna(0) #NaNs in the input cause problems

    #insert the dataframe without the background class and the variable for regression
    mlDataR = MlData(df,'HT') 

    mlDataR.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now

    #Now decide whether we want to standardise the dataset
    #it is worth seeing what happens to training with and without this option
    #(this must be done after the split to avoid information leakage)
    #mlDataR.standardise() #find this causes problems with regression

    print 'Setting up network'

    dnnR=Dnn(mlDataR,os.path.join(output,'regression'),doRegression=True)

    #here sets up with mean squared error and a linear output neuron
    dnnR.setup(hiddenLayers=[20,20],dropOut=None,l2Regularization=None)#,loss='mean_squared_error')

    print 'Fitting'
    dnnR.fit(epochs=100,batch_size=128,callbacks = [earlyStopping])

    print 'Making diagnostics'
    dnnR.diagnostics() #make regression specific diagnostics

if doHyperOpt:

    print 'Running Huper parametr opt'
    print '------Timer start--------'
    start_time = time.time()

    print 'Preparing data'
    mlDataC = MlData(df,'signal') #insert the dataframe and tell it what the truth variable is
    mlDataC.split(evalSize=0.0,testSize=0.3) #Split into train and test sets, leave out evaluation set for now 

    # print 'Data output before standardise:'
    # mlDataC.output(number_of_lines=5)

    mlDataC.standardise()

    print 'Data output after standardisation:'
    mlDataC.output(number_of_lines=5)

    print 'Defining XGB'
    bdt = Bdt(mlDataC,output+'/XGB',separation_facet=0.957)

    space4rf = {
        'max_depth':hp.choice('max_depth',range(1,10)),
        'min_child_weight':hp.choice('min_child_weight',range(1,20)),
        'gamma': hp.choice('gamma',[x * 0.1 for x in range(0, 8)]),
        # 'subsample' : hp.choice('subsample',[i/100.0 for i in range(60,100,5)]),
        # 'colsample_bytree' : hp.choice('colsample_bytree',[i/100.0 for i in range(60,100,5)]),
        # 'reg_alpha': hp.choice('reg_alpha',[1e-5, 1e-2, 0.1, 1, 100]),
        # 'eval_metric':hp.choice('eval_metric',['rmse','mae','logloss','error','merror','mlogloss','auc']),
        # 'eta': hp.choice('eta',[x * 0.01 for x in range(1, 20)]),
        # 'n_estimators': hp.choice('n_estimators',range(80,800)),
        # 'learning_rate': hp.choice('learning_rate',[0.001,0.01,0.1,0.5,1]),

    }

    trials = Trials()
    
    bdt.setup(cls=XGBClassifier,objective= 'binary:logistic', nthread=6, seed=27)
    
    # f = bdt.hyperopt_train_test(XGBClassifier())
    best = fmin(bdt.hyperopt_train_test_xgb, space4rf, algo=tpe.suggest, max_evals=100, trials=trials)

    print "best:"
    print best

    bdt.print_best_hyperopt_results()
