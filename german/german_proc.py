# 28 December 2019
#
# Processing german credit data set
#
# We might replace maritial status with something like relationship categories
# in adult (to allow us to compute spouse consistency)
# 
# The initial processing (get_german_data) function is based on the 
# AIF360 code german.py
# See github.com/IBM/AIF360

import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb
from copy import deepcopy

import sys

# path for boosting scripts
sys.path.append('../scripts')

import script


# See github.com/IBM/AIF360
from aif360.datasets import GermanDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV

filepath = 'PATH_TO_RAW_GERMAN_DATA/raw/german/'

def german_preprocessing(df):
    """
    Make a copy of the age attribute that we will binarize
    """

    #print(df['age'])
    df['age_bin'] = df['age']

    return df

default_mappings = {
    'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Old', 0.0: 'Young'}],
}

# Only consider age as protected
# Don't binarize it by default
# Keep personal status as categorical (don't convert to sex)
def get_german_data():

    dayta = GermanDataset(favorable_classes=[1],
                 protected_attribute_names=['age_bin'],
                 privileged_classes=[lambda x: x >= 25],
                 instance_weights_name=None,
                 categorical_features=['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'personal_status',
                     'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=german_preprocessing,
                 metadata=default_mappings)

    return dayta


def train_ridge(X_train, y_age_train, aind, abind=None, RCV=RidgeCV()):

    XR = np.copy(X_train)
    XR[:,aind] = 0
    if abind is not None:
        XR[:,abind] = 0

    RCV.fit(XR, y_age_train)

    return RCV

# We never want the binaray age to be part of the protected subspace.
# We will often get rid of it completely, but we never want it in 
# the protected subspace.
#
# Also, a terrible projection matrix generation function.  Oh well.
def german_proj_mat(RCV, aind):

    vec = RCV.coef_
    A = np.zeros((vec.shape[0], 2))
    A[:,0] = vec
    A[aind,1] = 1.

    return script.proj_matrix_gen( 
            np.zeros((vec.shape[0], vec.shape[0])),
            A,
            test=True
            )

# protected indices are age only
# My simple tests suggest that this appears to be working
def get_german_train_test_age(
        dataset_orig,
        pct=0.8,
        traininds=None,
        testinds=None,
        seed=None,
        removeProt=False,
        ):

    # we will standardize continous features
    continous_features = [
            'month',
            'credit_amount',
            'investment_as_income_percentage',
            'residence_since',
            'age',
            'number_of_credits',
            'people_liable_for'
        ]


    continous_features_indices = [
            dataset_orig.feature_names.index(feat) 
            for feat in continous_features
        ]

    print(continous_features_indices)

    X_full = dataset_orig.features
    y_full = dataset_orig.labels.T[0] - 1
    train_size = np.floor( pct*X_full.shape[0] ).astype('int')

    if traininds is None or testinds is None:
        splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=train_size,
                random_state=seed
                )
        gen = splitter.split(X_full, y_full)
        for train, test in gen:
            traininds = train
            testinds = test

    # make more or less than 50k per year
    X_train = X_full[traininds]
    y_train = y_full[traininds]

    X_test = X_full[testinds]
    y_test = y_full[testinds]

    aind = dataset_orig.feature_names.index('age')
    abind = dataset_orig.feature_names.index('age_bin')
    y_age_train = X_train[:, aind]
    y_age_test = X_test[:, aind]

    ### PROCESS TRAINING DATA
    # normalize continuous features
    SS = StandardScaler().fit(X_train[:, continous_features_indices])
    X_train[:, continous_features_indices] = SS.transform(
            X_train[:, continous_features_indices]
    )

    # remove age as predictive features
    # We don't do this for the data in the submitted version
    #
    # Remember that this messes up your feature_names
    if removeProt:
        X_train = np.delete(
                X_train, 
                [ aind, abind], 
                axis = 1
        )

    ### PROCESS TEST DATA
    # normalize continuous features
    X_test[:, continous_features_indices] = SS.transform(
            X_test[:, continous_features_indices]
    )

    # remove age as predictive features
    if removeProt:
        X_test = np.delete(
                X_test, 
                [ aind, abind ], 
                axis = 1
        )

    return X_train, X_test, y_train, y_test, y_age_train, y_age_test, dataset_orig.feature_names, traininds, testinds


# protected indices are age and sex
def get_german_train_test(
        dataset_orig,
        pct=0.8,
        traininds=None,
        testinds=None,
        seed=None,
        removeProt=True,
        binary_age=True
        ):

    # we will standardize continous features
    continous_features = [
            'month',
            'credit_amount',
            'investment_as_income_percentage',
            'residence_since',
            'number_of_credits',
            'people_liable_for'
        ]

    if not binary_age: continuous_features += ['age']

    continous_features_indices = [
            dataset_orig.feature_names.index(feat) 
            for feat in continous_features
        ]

    print(continous_features_indices)

    X_full = dataset_orig.features
    y_full = dataset_orig.labels.T[0] - 1
    train_size = np.floor( pct*X_full.shape[0] ).astype('int')

    if traininds is None or testinds is None:
        splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=train_size,
                random_state=seed
                )
        gen = splitter.split(X_full, y_full)
        for train, test in gen:
            traininds = train
            testinds = test

    # make more or less than 50k per year
    X_train = X_full[traininds]
    y_train = y_full[traininds]

    X_test = X_full[testinds]
    y_test = y_full[testinds]

    sind = dataset_orig.feature_names.index('sex')
    aind = dataset_orig.feature_names.index('age')
    y_sex_train = X_train[:, sind]
    y_sex_test = X_test[:, sind]
    y_age_train = X_train[:, aind]
    y_age_test = X_test[:, aind]

    ### PROCESS TRAINING DATA
    # normalize continuous features
    SS = StandardScaler().fit(X_train[:, continous_features_indices])
    X_train[:, continous_features_indices] = SS.transform(
            X_train[:, continous_features_indices]
    )

    # remove sex and age as predictive features
    # But not we are actually going to do it.
    if removeProt:
        X_train = np.delete(
                X_train, 
                [ sind, aind ], 
                axis = 1
        )

    # find sensitive directions
    mu0 = X_train.mean(axis=0)
    mu11 = X_train[ y_sex_train*y_age_train > 0 ].mean(axis=0)
    mu10 = X_train[ (1 - y_sex_train)*y_age_train > 0].mean(axis=0)
    mu01 = X_train[ y_sex_train*(1 - y_age_train) > 0].mean(axis=0)
    mu00 = X_train[ (1-y_sex_train)*(1-y_age_train) > 0].mean(axis=0)
    A = np.array([ mu11, mu10, mu01, mu00 ])
    A = A - mu0

    ### PROCESS TEST DATA
    # normalize continuous features
    X_test[:, continous_features_indices] = SS.transform(
            X_test[:, continous_features_indices]
    )

    # remove sex and age as predictive features
    if removeProt:
        X_test = np.delete(
                X_test, 
                [ sind, aind ], 
                axis = 1
        )

    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_age_train, y_age_test, dataset_orig.feature_names, A, traininds, testinds

# Import the one-hot encoded data here or not whatever
# Good: xgb.train(param, dtrain, 1000, watchlist). Gets balanced accuracy of ~84%
# for adult
def prep_baseline_xgb(X_train, X_test, y_train, y_test, param=None):

    y_train_real = y_train.astype('int')
    y_test_real = y_test.astype('int')
    if (len(y_train.shape) > 1):
        y_train_real = y_train[:,1].copy().astype('int')
        y_test_real = y_test[:,1].copy().astype('int')

    dtrain = xgb.DMatrix(data=X_train, label=y_train_real)
    dtest = xgb.DMatrix(data=X_test, label=y_test_real)

    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    # Have only done some quick testing here.  So good stuff.
    num1 = y_train_real.sum()
    num0 = (1-y_train_real).sum()
    if param is None:
        param = {
                'max_depth':5,
                'eta':0.05,
                'objective':'binary:logistic',
                'min_child_weight':0.5,
                'lambda':0.0001,
                'scale_pos_weight':num0/num1
        }

    else:
        if 'scale_pos_weight' in param:
            param['scale_pos_weight'] += num0/num1
        else: 
            param['scale_pos_weight'] = num0/num1

    return dtrain, dtest, watchlist, param, num0, num1



def sex_age_cons(bst, X_test, sind, aind):

    # Only make one copy for SPEED
    X = deepcopy(X_test)

    X[:, sind] = np.zeros(X.shape[0])
    X[:, aind] = np.zeros(X.shape[0])
    preds00 = bst.predict(xgb.DMatrix(X))
    y00 = np.array([ 1 if pval > 0.5 else 0 for pval in preds00])

    X[:, sind] = np.zeros(X.shape[0])
    X[:, aind] = np.ones(X.shape[0])
    preds01 = bst.predict(xgb.DMatrix(X))
    y01 = np.array([ 1 if pval > 0.5 else 0 for pval in preds01])

    X[:, sind] = np.ones(X.shape[0])
    X[:, aind] = np.zeros(X.shape[0])
    preds10 = bst.predict(xgb.DMatrix(X))
    y10 = np.array([ 1 if pval > 0.5 else 0 for pval in preds10])

    X[:, sind] = np.ones(X.shape[0])
    X[:, aind] = np.ones(X.shape[0])
    preds11 = bst.predict(xgb.DMatrix(X))
    y11 = np.array([ 1 if pval > 0.5 else 0 for pval in preds11])

    # Want all quadruples to be consistent (all 0s or all 1s)
    return (y00*y01*y10*y11).sum() + ((1-y00)*(1-y01)*(1-y10)*(1-y11)).sum()


def status_cons(bst, X_test, status_inds):

    # Only make one copy for SPEED
    X = deepcopy(X_test)

    guesses = np.zeros((X.shape[0], len(status_inds)))

    for i, ind in enumerate(status_inds):

        X[:,status_inds] = 0.
        X[:,ind] = 1.

        preds = bst.predict(xgb.DMatrix(X))
        guesses[:,i] = np.array([ 1 if pval > 0.5 else 0 for pval in preds])

    cons = guesses.sum(axis=1)
    comp_val = len(status_inds)

    return np.logical_or(cons==comp_val, cons==0) 

def flip_cons(bst, X_test, ind):

    # Only make one copy for SPEED
    X = deepcopy(X_test)

    guesses = np.zeros((X.shape[0], 2))

    X[:,ind]=0.0
    preds = bst.predict(xgb.DMatrix(X))
    guesses[:,0] = np.array([ 1 if pval > 0.5 else 0 for pval in preds])

    X[:,ind]=1.0
    preds = bst.predict(xgb.DMatrix(X))
    guesses[:,1] = np.array([ 1 if pval > 0.5 else 0 for pval in preds])

    cons = guesses.sum(axis=1)

    return np.logical_or(cons==2, cons==0)


# only considering scale_pos_weight default right now
def collect_xgb_fairness_data(seeds, params, n_iter, project=False, agebin=False, filename=None):
      
    dayta = get_german_data()

    # Collect p0, p1, balacc, age TPR/TNR, age GAPS, 3x group metrics
    # sex TPR/TNR, sex GAPS, 3x group metrics
    # Potentially two consistencies:
    # If we have binary age - want to flip it
    # Otherwise, look at consistency among all four types of personal status
    finfo = np.zeros((seeds.shape[0], 21))
    if agebin:
        consinfo = np.zeros((seeds.shape[0],2))
    else:
        consinfo = np.zeros((seeds.shape[0], 1))
    
    for iseed, seed in enumerate(seeds):
        print("Collecting tha dayta! (seed: {})".format(seed))

        (
                X_train,
                X_test,
                y_train,
                y_test,
                y_age_train,
                y_age_test,
                feature_names,
                traininds,
                testinds
        ) = get_german_train_test_age(
                dayta,
                pct=0.8,
                removeProt=False,
                seed=seed
                )

        aind = feature_names.index('age')
        abind = feature_names.index('age_bin')

        # Get status_inds and binary gender labels
        status_inds = []
        y_sex_bin = np.zeros(X_train.shape[0])
        yt_sex = np.zeros(X_test.shape[0])

        for i, nom in enumerate(feature_names):
            if "personal_status" in nom:
                status_inds.append(i)

                # Males are 1, females are 0 in the following
                if 'A92' not in nom and 'A95' not in nom:
                    y_sex_bin += X_train[:,i]
                    yt_sex += X_test[:,i]

        print("Personal status indices: {}".format(status_inds))
        muppet = y_sex_bin.max()
        if muppet > 1: 
            print("Warning: in train set, you messed up the gender binary")
        muppet = yt_sex.max()
        if muppet > 1: 
            print("Warning: in test set, you messed up the gender binary")

        # Get the binary age to test against now
        y_age_bin = np.copy(X_train[:,abind])
        yt_age = np.copy(X_test[:,abind])

        if not agebin:
            print("Eliminating binary age attribute")
            X_train[:,abind] = 0
            X_test[:, abind] = 0

        if project:
            print("Projecting tha dayta!")
            RCV = train_ridge(X_train, y_age_train, aind=aind, abind=None)
            proj = german_proj_mat(RCV, aind)
            X_train = np.matmul(X_train, proj)
            X_test = np.matmul(X_test, proj)

        res = dict()
        dtrain, dtest, watchlist, param, _, _ = prep_baseline_xgb(X_train, X_test, y_train, y_test)

        param['max_depth'] = params['max_depth']
        param['eta'] = params['eta']
        param['min_child_weight'] = params['min_child_weight']
        param['lambda'] = params['lambda']

        bst = xgb.train(
                param,
                dtrain,
                n_iter,
                evals=watchlist,
                evals_result=res,
                verbose_eval=False
                )

        predst = bst.predict(dtest)
        y_guess = np.array([1 if pval > 0.5 else 0 for pval in predst])

        # No information about training... not even sure I want these
        p0, p1, _ = script.balanced_accuracy(bst, dtest, y_test)

        # consistency measures...
        scons = status_cons(bst, X_test, status_inds).sum()
        consinfo[iseed, 0] = scons/X_test.shape[0]
        if agebin:
            abcons = flip_cons(bst, X_test, abind).sum()
            consinfo[iseed, 1] = abcons/X_test.shape[0]

        print("GENDER INFO")
        sex_res = script.group_metrics(
                y_test,
                y_guess,
                yt_sex,
                label_good=1,
                verbose=True
                )

        print("AGE INFO")
        age_res = script.group_metrics(
                y_test,
                y_guess,
                yt_age,
                label_good=1,
                verbose=True
                )

        finfo[iseed,0] = p0
        finfo[iseed,1] = p1
        finfo[iseed,2] = (p0 + p1)/2

        finfo[iseed, range(3,12)] = age_res
        finfo[iseed, range(12,21)] = sex_res

    # DONE LOOP

    # For now, just save the arrays.
    # Maybe I can be nicer and provide column names in the future
    if filename is None:
        filename = "fairness-metrics"
        if agebin: filename += '-agebin'
        if project: filename += '-project'
        filename += '.npz'

    np.savez(filename, finfo=finfo, consinfo=consinfo)
    

    """
    # Old way to save the information
    f.write('test-balanced\t{}\n'.format( (p0+p1)/2 ))
    f.write('p0\t{}\n'.format(p0))
    f.write('p1\t{}\n'.format(p1))

    f.write('age-TPR-prot\t{}\n'.format(age_res[0]))
    f.write('age-TNR-prot\t{}\n'.format(age_res[1]))
    f.write('age-TPR-priv\t{}\n'.format(age_res[2]))
    f.write('age-TNR-priv\t{}\n'.format(age_res[3]))
    f.write('age-gap-RMS\t{}\n'.format(age_res[4]))
    f.write('age-gap-MAX\t{}\n'.format(age_res[5]))
    f.write('age-ave-odds-diff\t{}\n'.format(age_res[6]))
    f.write('age-eq-opp-diff\t{}\n'.format(age_res[7]))
    f.write('age-stat-parity\t{}\n'.format(age_res[8]))
    """
