# Python Module of XGBoost for JMP
# Please send comments, suggestions, corrections to russ.wolfinger@jmp.com

# make sure all of these modules are installed before running
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc

#%%
# replace special characters in feature names
def feature_name_replace(f):
    f = ''.join(f)
    # print(i,f)
    f = f.replace('(','_')
    f = f.replace(')','_')
    f = f.replace('[','_')
    f = f.replace(']','_')
    f = f.replace('{','_')
    f = f.replace('}','_')
    f = f.replace(' ','_')
    f = f.replace('.','_')
    f = f.replace("'",'_')
    f = f.replace(',','_')
    return f  

# write a feature map file
def create_feature_map(features, fname):
    outfile = open(fname, 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
    # print(fname)
  

#%%
# main function to be called after input variables are configured
# mname: prefix name of model
# path: working directory
# df: pandas dataframe of input data
# yvar: names of response variables paired with their objective functions in a dictionary
# xvar: names of predictor variables as a list, categorical variables are automatically one-hot encoded
# wvar: name of weight variable in a list or empty list for no weight
# vvar: validation variables whose levels correspond to folds
# cvar: categorical variables as a dictionary specifying levels as a list for each variable
# parms: xgboost parameters, only those different from defaults are needed
# iter: number of iterations for training
# action='fit': fit xgboost models with repeated k-fold cross validation
# action='predict': predict new data, binary model objects must already be saved in mod subdirectory
# versbose_eval: how frequently to print xgboost iterations
# plot: plot actual versus predicted when action=='fit'
# returns: pandas dataframe of predictions
def main(mname, path, df, yvar, xvar, wvar, vvar, cvar, parms, iter, action='fit',
         verbose_eval=100, verbose=True, plot=True):

    mname0 = mname
    if verbose:
        print('mname',mname0)
    
    nv = len(vvar)
    if nv == 0: vvar = [0]
    
    PATH_MOD = path + 'mod/'
    os.makedirs(PATH_MOD, exist_ok=True)
    
    if action=='fit':
        PATH_OOF = path + 'oof/'
        PATH_IMP = path + 'imp/'
        os.makedirs(PATH_OOF, exist_ok=True)
        os.makedirs(PATH_IMP, exist_ok=True)
           
    yvk = list(yvar.keys())
    
    # loop over y variables
    plist = []
    for yvi, yv in enumerate(yvk):
    
        parms['objective'] = yvar[yv]
        if parms['objective'] in ['multi:softprob'] :
            nl = len(cvar[yv])
            parms['num_class'] = nl
        else:
            if 'num_class' in parms: del parms['num_class']
            nl = 1
            
        binary_obj = parms['objective'][:6] == 'binary'
        
        # for training, drop rows with missing y values
        if action=='fit':
            # fixme: check if this works for character y values with blank missing
            m = df.dropna(subset=yvk)
            
            y = m[yv].values
            
            # label encode nominal response
            if yv in cvar:
                le = preprocessing.LabelEncoder()
                le.fit(cvar[yv])
                y = le.transform(y) 
            if verbose:    
                print('y.shape',y.shape)
        else:
            m = df
        
        if verbose:
            print(m.shape)
        
        
        #%%
        # one-hot encode variables in x, preserving order
        # fixme: check cat variables with missing values, verify c++ code matches
        catx = False
        nx = 0
        features = []
        for x in xvar: 
            if x in cvar: 
                catx = True
                lc = len(cvar[x])
                # drop first column for 2-level effects
                if (lc == 2): lc = 1
                nx += lc
            else:
                nx += 1
                
        if catx:
            X = np.zeros((m.shape[0],nx))
            n = 0
            for xi, x in enumerate(xvar): 
                if x in cvar: 
                    lc = len(cvar[x])
                    # drop first column for 2-level effects
                    if (lc == 2): 
                        lc = 1
                        drop_first = True
                    else:
                        drop_first = False
                    mx = pd.DataFrame(m[x]).astype('category',categories=cvar[x])
                    X[:,n:(n+lc)] = pd.get_dummies(mx, prefix=x, drop_first=drop_first).values
                    n += lc
                    for i,c in enumerate(cvar[x]):
                        if (i==0 and drop_first): continue
                        features.append(x + '_' + str(c))
                else:
                    X[:,n] = m[x].values
                    n += 1
                    features.append(x)
            
        else:
            X = m[xvar].values
            features = xvar
        
        if action=='fit':
            if verbose:
                print('X.shape',X.shape)
                print('features',features)
            
            # setup for feature importance
            f0 = features
            nf = len(f0)
            if verbose:
                print('number of features = ' + str(nf))
                
            # replace special characters, xgboost variable importance does not like them
            features = []
            for i in range(nf):
                f = feature_name_replace(str(f0[i]))
                # print(i,f)
                features.append(f)
            fmapname = PATH_IMP + mname + '.fmap'
            create_feature_map(features,fname=fmapname)
            if verbose:
                print(fmapname)
            
        #%%
        
        # total number of runs
        nfit = 0
        if nv:
            for v in vvar:
                if (v in cvar): vf = cvar[v]
                else: vf = [0]
                nfit += len(vf)
        else:
            nfit = 1
        if verbose:
            print('number of fits = ' + str(nfit))
        
        # arrays to save predictions across all fits
        train_pred = np.zeros((m.shape[0], nl, nfit))
        if nv and action=='fit':
            valid_pred = np.zeros((m.shape[0], nl, nfit))
        
        # fixme average over all importances
                
        # loop over validation variables and folds
        nr = 0
        for vi, v in enumerate(vvar):
         
            if nv: 
                vval = m[v].values
                if (v in cvar): vf = cvar[v]
                else: vf = [0]
            else:
                vf = [0]
        
            # loop over folds
            for fi, f in enumerate(vf):
                
                gc.collect();
        
                # if f != fold: continue
        
                mname = mname0 + 'v' + str(vi+1) + 'f' + str(fi+1)
                if len(yvk) > 1: mname += 'y' + str(yvi+i)
                
                if verbose:
                    print('')
                    print('*'*30)
                    print(mname)
                    print('*'*30)
                        
                if nv and action=='fit':
                    train_X = X[vval != f]
                    train_y = y[vval != f]
                    valid_X = X[vval == f]
                    valid_y = y[vval == f]
                else:
                    train_X = X
                    if action=='fit':
                        train_y = y
                        
                if action=='fit':   
                    if verbose:
                        print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)
                    train_d = xgb.DMatrix(train_X, train_y, missing=np.nan)
                else:
                    train_d = xgb.DMatrix(train_X, missing=np.nan)
                
                fname = PATH_MOD + mname + '.model'
                
                if action=='fit':
                    if nv: 
                        valid_d = xgb.DMatrix(valid_X, valid_y, missing=np.nan)
                        evals = [(train_d, 'train'), (valid_d, 'val')]
                    else:
                        evals = [(train_d, 'train')]
                           
                    # train
                    xg = xgb.train(parms, train_d, num_boost_round=iter, evals=evals,
                                       verbose_eval=verbose_eval)
                    
                    # save model binary object
                    # xg.save_model(fname)
                    # print(fname)
                else:
                    # load model binary object
                    xg = xgb.Booster()
                    xg.load_model(fname)
               
                pred = xg.predict(train_d)
                if len(pred.shape) == 1: pred = pred.reshape(-1,1)
                
                if nv and action=='fit': 
                    train_pred[vval != f, :, nr] += pred
                    poof = xg.predict(valid_d)        
                    if len(poof.shape) == 1: poof = poof.reshape(-1,1)
                    valid_pred[vval == f, :, nr] += poof
                else:
                    train_pred[:, :, nr] += pred
                       
                # variable importance
                if action=='fit':
                    gainf = xg.get_score(fmap=fmapname,importance_type='gain')
                    splitf = xg.get_score(fmap=fmapname,importance_type='weight')   
                    gainf = pd.DataFrame.from_dict(gainf,orient='index')
                    splitf = pd.DataFrame.from_dict(splitf,orient='index')
                    gainf.columns = ['gainf']
                    splitf.columns = ['splitf']
            
                    imp = pd.DataFrame({'feature':features,'gain':0.0,'split':0.0})
                    imp.set_index(['feature'],inplace=True)
            
                    imp = imp.merge(gainf,how='left',left_index=True,right_index=True)
                    imp = imp.merge(splitf,how='left',left_index=True,right_index=True)
                    imp.gainf.fillna(0,inplace=True)
                    imp.splitf.fillna(0,inplace=True)
                    imp.gain += imp.gainf
                    imp.split += imp.splitf
                    imp = imp.drop(['gainf','splitf'],axis=1)
                    imp.gain /= sum(imp.gain)
                    imp.split /= sum(imp.split)
                    imp.sort_values(['gain'],ascending=False,inplace=True)
            
                    os.makedirs('imp',exist_ok=True)
                    fname = PATH_IMP+mname+'.csv'
                    imp.to_csv(fname)
                    if verbose:
                        print('')
                        print(imp.head(n=3))
                        print(fname)
                    
                    nr += 1;
                    
           
        #    if nv:
        #        mname = mname0 + 'v' + str(vi+1)
        #        vp = valid_pred[:,nr]
        #        print(vp.shape, vp.min(), vp.mean(), vp.max())
        #        sns.distplot(vp)
        #        plt.title("Distribution of Out-of-Fold Predictions for " + mname)
        #        plt.show()
        #        plt.gcf().clear()
        
        #    print(mname,'oof corr',corr(y,oofs))
                
        
        #%%
        if verbose:
            print('')
            print(yv, mname0, parms)
        
        # compute prediction averages
        train_pred[train_pred==0] = np.nan
        train_pred_avg = np.nanmean(train_pred, axis=2)
        
        if action=='fit':
            if plot:
                # compute prob actual for nominal y
                if yv in cvar:
                    if binary_obj:
                      train_pred_actual = train_pred_avg[:,0]
                      train_pred_actual[y==0] = 1.0 - train_pred_actual[y==0]
                    else:
                      train_pred_actual = train_pred_avg[np.arange(y.shape[0]),y]
                else:
                    train_pred_actual = train_pred_avg[:,0]
                
                # plot actual by predicted 
                if nv:
                    valid_pred[valid_pred==0] = np.nan
                    valid_pred_avg = np.nanmean(valid_pred, axis=2)
                    
                    # compute prob actual for nominal y
                    if yv in cvar:
                        if binary_obj:
                          valid_pred_actual = valid_pred_avg[:,0]
                          valid_pred_actual[y==0] = 1.0 - valid_pred_actual[y==0]
                        else:
                          valid_pred_actual = valid_pred_avg[np.arange(y.shape[0]),y]
                    else:
                        valid_pred_actual = valid_pred_avg[:,0]
                    
                    
                    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12,5))
                    sns.scatterplot(x=train_pred_actual, y=y, ax=ax1)
                    sns.scatterplot(x=valid_pred_actual, y=y, ax=ax2)
                
                    fig.suptitle(yv + ' ' + mname0 + ' Actual by Predicted')
                    ax1.set_title('Training')
                    ax2.set_title('Validation')
                    plt.show()
                    plt.gcf().clear()
                      
                else:
                    sns.scatterplot(y, train_pred_actual)
                    plt.title(mname0 + ' Actual vs. Predicted Training')
                    plt.show()
                    plt.gcf().clear()
                
            if nv: 
                # use m to preserve index for later merging
                vp = m[[]].copy()
                vp[mname0] = valid_pred_avg
                # save average out of fold predictions
                fname = PATH_OOF + mname0 + '.csv'
                vp.to_csv(fname, index=False)
                if verbose:
                    print(fname, valid_pred.shape)
                plist.append(vp)
            else: 
                # use m to preserve index for later merging
                tp = m[[]].copy()
                tp[mname0] = train_pred_avg
                plist.append(tp)
                
        else: 
            tp = m[[]].copy()
            tp[mname0] = train_pred_avg
            plist.append(tp)
    
    if len(plist) == 1: return plist[0]
    else: return plist
            

def fit(mname, path, data, yvar, xvar, wvar, vvar, cvar, parms, iter, verbose_eval=100,
        verbose=True, plot=True):
   vp = main(mname, path, data, yvar, xvar, wvar, vvar, cvar, parms, iter, action='fit',
               verbose_eval=verbose_eval, verbose=verbose, plot=plot)
   return vp

# weight, parms, iter are not needed for prediction
# data values for yvar are also not needed, just names and objective function pairs
def predict(mname, path, data, yvar, xvar, vvar, cvar):
   tp = main(mname, path, data, yvar, xvar, [], vvar, cvar, {}, 0, action='predict',
             verbose=False, plot=False)
   return tp

