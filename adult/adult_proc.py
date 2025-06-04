# November 26, 2019
## Processing the adult data set to output a usuable train/test split.
# 
# All data for loading adult data from the original SenSR repository.
# github.com/IBM/sensitive-subspace-robustness
#
# Also need AIF360
# github.com/IBM/AIF360

from aif360.datasets import BinaryLabelDataset, AdultDataset

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from copy import deepcopy

import script
import numpy as np
import scipy as sp
import tensorflow as tf

import xgboost as xgb

from sklearn.linear_model import LogisticRegressionCV

def get_adult_orig():

    dataset_orig = AdultDataset()

    # We do not use these features. Note, we use the continuous version of
    # education, i.e. `education-num`, so we drop the categorical versions of
    # education
    drop_features = [
        'education=10th',
        'education=11th',
        'education=12th',
        'education=1st-4th',
        'education=5th-6th',
        'education=7th-8th',
        'education=9th',
        'education=Assoc-acdm',
        'education=Assoc-voc',
        'education=Bachelors',
        'education=Doctorate',
        'education=HS-grad',
        'education=Masters',
        'education=Preschool',
        'education=Prof-school',
        'education=Some-college', 
        'native-country=Cambodia',
        'native-country=Canada',
        'native-country=China',
        'native-country=Columbia',
        'native-country=Cuba',
        'native-country=Dominican-Republic',
        'native-country=Ecuador',
        'native-country=El-Salvador',
        'native-country=England',
        'native-country=France',
        'native-country=Germany',
        'native-country=Greece',
        'native-country=Guatemala',
        'native-country=Haiti',
        'native-country=Holand-Netherlands',
        'native-country=Honduras',
        'native-country=Hong',
        'native-country=Hungary',
        'native-country=India',
        'native-country=Iran',
        'native-country=Ireland',
        'native-country=Italy',
        'native-country=Jamaica',
        'native-country=Japan',
        'native-country=Laos',
        'native-country=Mexico',
        'native-country=Nicaragua',
        'native-country=Outlying-US(Guam-USVI-etc)',
        'native-country=Peru',
        'native-country=Philippines',
        'native-country=Poland',
        'native-country=Portugal',
        'native-country=Puerto-Rico',
        'native-country=Scotland',
        'native-country=South',
        'native-country=Taiwan',
        'native-country=Thailand',
        'native-country=Trinadad&Tobago',
        'native-country=United-States',
        'native-country=Vietnam',
        'native-country=Yugoslavia']

    drop_features_indices = [
            dataset_orig.feature_names.index(feat) 
            for feat in drop_features
        ]

    dataset_orig.features = np.delete(
            dataset_orig.features, 
            drop_features_indices, 
            axis = 1
    )

    dataset_orig.feature_names = [
            feat 
            for feat in dataset_orig.feature_names 
            if feat not in drop_features
        ]

    return dataset_orig

# Split the dataset and split into train and test
def get_adult_train_test_sensr(dataset_orig, pct=0.8):

    # we will standardize continous features
    continous_features = [
            'age', 
            'education-num', 
            'capital-gain', 
            'capital-loss', 
            'hours-per-week'
        ]
    continous_features_indices = [
            dataset_orig.feature_names.index(feat) 
            for feat in continous_features
        ]

    # get a train/test split with the input percentage (default = 0.8)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([pct], shuffle=True)

    X_train = dataset_orig_train.features

    # find sensitive directions
    mu0 = X_train.mean(axis=0)

    # normalize continuous features
    SS = StandardScaler().fit(X_train[:, continous_features_indices])
    X_train[:, continous_features_indices] = SS.transform(
            X_train[:, continous_features_indices]
    )

    # remove sex and race as predictive features
    # We don't want to do this when we are looking at the ICLR submission version
    # But not we are actually going to do it.
    X_train = np.delete(
            X_train, 
            [ 
                dataset_orig_train.feature_names.index(feat) 
                for feat in ['sex', 'race']
            ], 
            axis = 1
    )

    X_test = dataset_orig_test.features

    # normalize continuous features
    X_test[:, continous_features_indices] = SS.transform(
            X_test[:, continous_features_indices]
    )

    # remove sex and race as predictive features
    X_test = np.delete(
            X_test, 
            [
                dataset_orig_test.feature_names.index(feat) 
                for feat in ['sex', 'race']
            ], 
            axis = 1
    )

    # make more or less than 50k per year
    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels

    # One hot encoding of labels
    one_hot = OneHotEncoder(sparse=False)
    one_hot.fit(y_train.reshape(-1,1))
    y_train = one_hot.transform(y_train.reshape(-1,1))
    y_test = one_hot.transform(y_test.reshape(-1,1))

    y_sex_train = dataset_orig_train.features[
            :, 
            dataset_orig_train.feature_names.index('sex')
        ]
    y_sex_test = dataset_orig_test.features[
            :, 
            dataset_orig_test.feature_names.index('sex')
        ]
    one_hot.fit(y_sex_train.reshape(-1,1))
    y_sex_train = one_hot.transform(y_sex_train.reshape(-1,1))
    y_sex_test = one_hot.transform(y_sex_test.reshape(-1,1))

    y_race_train = dataset_orig_train.features[
            :, 
            dataset_orig_train.feature_names.index('race')
        ]
    y_race_test = dataset_orig_test.features[
            :, 
            dataset_orig_test.feature_names.index('race')
        ]
    one_hot.fit(y_sex_train.reshape(-1,1))
    y_race_train = one_hot.transform(y_race_train.reshape(-1,1))
    y_race_test = one_hot.transform(y_race_test.reshape(-1,1))


    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, dataset_orig_train.feature_names

def load_adult_train_test_sensr(
        filename='sensr_data',
        seed=None
        ):

    print("Loading data from file with seed " + str(seed), flush=True)

    saved_data = np.load(filename + '_' + str(seed) + '.npz')

    X_train = saved_data['X_train']
    X_test = saved_data['X_test']

    y_train = saved_data['y_train'][:,0]
    y_test = saved_data['y_test'][:,0]

    feature_names = list(saved_data['feature_names'])
    A = saved_data['proj']

    # change the feature names to be consistent with the names that
    # we have been working with
    sind = feature_names.index('sex_ Male')
    rind = feature_names.index('race_ White')

    feature_names[sind] = 'sex'
    feature_names[rind] = 'race'

    y_sex_train = X_train[:,sind]
    y_sex_test = X_test[:,sind]

    y_race_train = X_train[:,rind]
    y_race_test = X_test[:,rind]

    traininds = None
    testinds = None

    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, A, traininds, testinds

def get_adult_train_test(
        dataset_orig,
        pct=0.8,
        traininds=None,
        testinds=None,
        seed=None,
        removeProt=True,
        ):

    # we will standardize continous features
    continous_features = [
            'age', 
            'education-num', 
            'capital-gain', 
            'capital-loss', 
            'hours-per-week'
        ]
    continous_features_indices = [
            dataset_orig.feature_names.index(feat) 
            for feat in continous_features
        ]

    X_full = dataset_orig.features
    y_full = dataset_orig.labels.T[0]
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
    rind = dataset_orig.feature_names.index('race')
    y_sex_train = X_train[:, sind]
    y_sex_test = X_test[:, sind]
    y_race_train = X_train[:, rind]
    y_race_test = X_test[:, rind]

    ### PROCESS TRAINING DATA
    # normalize continuous features
    SS = StandardScaler().fit(X_train[:, continous_features_indices])
    X_train[:, continous_features_indices] = SS.transform(
            X_train[:, continous_features_indices]
    )

    # remove sex and race as predictive features
    # We don't want to do this when we are looking at the ICLR submission version
    # But not we are actually going to do it.
    if removeProt:
        X_train = np.delete(
                X_train, 
                [ sind, rind ], 
                axis = 1
        )

    # find sensitive directions
    mu0 = X_train.mean(axis=0)
    mu11 = X_train[ y_sex_train*y_race_train > 0 ].mean(axis=0)
    mu10 = X_train[ (1 - y_sex_train)*y_race_train > 0].mean(axis=0)
    mu01 = X_train[ y_sex_train*(1 - y_race_train) > 0].mean(axis=0)
    mu00 = X_train[ (1-y_sex_train)*(1-y_race_train) > 0].mean(axis=0)
    A = np.array([ mu11, mu10, mu01, mu00 ])
    A = A - mu0

    ### PROCESS TEST DATA
    # normalize continuous features
    X_test[:, continous_features_indices] = SS.transform(
            X_test[:, continous_features_indices]
    )

    # remove sex and race as predictive features
    if removeProt:
        X_test = np.delete(
                X_test, 
                [ sind, rind ], 
                axis = 1
        )

    # Not one-hot encoding here.  See the sensr code for how that works.

    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, dataset_orig.feature_names, A, traininds, testinds

def adult_feature_logreg(
        X_train,
        X_test=None,
        sind=None,
        feature_names=None,
        test=True,
        labels=None
        ):

    if feature_names and sind:
        print("Fitting LR to feature {}".format(feature_names[sind]))

    LR = LogisticRegressionCV(Cs=100, cv=5, max_iter=5000)

    # data for training logistic regression
    if X_test is not None:
        XLR = np.vstack((X_train, X_test))
    else:
        XLR = np.copy(X_train)

    if sind is not None: 
        targets = XLR[:,sind].copy()
        XLR[:,sind] = np.zeros(XLR.shape[0])

    elif labels is not None: 
        targets = labels

    else:
        print("Error: no labels provided for logistic regression")
        print("Error: no model trained", flush=True)
        return LR

    LR.fit( XLR, targets )

    if test and X_test is not None:
        outputs = LR.predict(np.vstack((X_train,X_test)))
        print("Training error of LR classifier: {}".format(
                np.abs(outputs-targets).sum()/XLR.shape[0]
            ))

    return LR

# projection onto sensitive subspace
def adult_proj_matrix_sensr(X_train, sind, feature_names, test=False,save=False):

    eg = np.zeros(X_train.shape[1])
    eg[feature_names.index('sex')] = 1.0

    er = np.zeros(X_train.shape[1])
    er[feature_names.index('race')] = 1.0

    TLR = adult_feature_logreg(X_train, sind=sind, test=test)
    wg = TLR.coef_[0]

    A = np.array([wg, eg, er]).T

    return script.proj_matrix_gen(X_train, A, test, save)


# Import the one-hot encoded data here or not whatever
# Good: xgb.train(param, dtrain, 1000, watchlist). Gets balanced accuracy of ~84%
def adult_prep_baseline_xgb(X_train, X_test, y_train, y_test, param=None):

    y_train_real = y_train.astype('int')
    y_test_real = y_test.astype('int')
    if (len(y_train.shape) > 1):
        y_train_real = y_train[:,1].copy().astype('int')
        y_test_real = y_test[:,1].copy().astype('int')

    dtrain = xgb.DMatrix(data=X_train, label=y_train_real)
    dtest = xgb.DMatrix(data=X_test, label=y_test_real)

    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    # Looks like we get ~87% test accuracy with these parameters.
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

# we take floor of the percent * size for train_size
# for a subsample X_train of 10000, take pct=.22114 on adult
def adult_setup(
        pct=0.8, 
        baseline=True, 
        nsteps=1000, 
        seed=None,
        param=None,
        removeProt=False,
        loadData=False,
        fileName=None
        #dtype='float32', 
        #tdtype=tf.float32
        ):

    # pull in the adult data
    if loadData:
        if fileName is None: fileName = 'sensr_data'

        X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, A, traininds, testinds = load_adult_train_test_sensr(filename=fileName, seed=seed)

    else: 
        orig_dataset = get_adult_orig()

        X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, A, traininds, testinds = get_adult_train_test(orig_dataset, pct=pct, seed=seed, removeProt=removeProt)

    # train the baseline
    dtrain, dtest, watchlist, param, _, _ = adult_prep_baseline_xgb(X_train, X_test, y_train, y_test, param=param)

    # project the training data and compute pairwise distances.
    # Make the required tensorflow constant array C
    if loadData:
        proj = A
        projData = np.matmul(X_train, proj)

    elif not removeProt:
        sind = feature_names.index('sex')
        proj = adult_proj_matrix_sensr(
            X_train, sind, feature_names, test=False,save=False
            )
        projData = np.matmul(X_train, proj)

    else:
        # Attempt to load a saved matrix from a while ago.
        # This case is not really completed.
        try:
            proj = np.load("adult_mat_small.npz")['mat']
            projData = np.matmul(X_train, proj)
        except:
            print("Sorry, your setup is screwed up right now :)")
            return

    Corig = sp.spatial.distance.squareform(
            sp.spatial.distance.pdist(
                projData,
                metric='sqeuclidean'
            ))#.astype(dtype)

    #C = tf.constant(Corig, dtype=tdtype)

    bst=None
    if baseline:
        bst = xgb.train(param, dtrain, nsteps, watchlist)

    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, Corig, dtrain, dtest, watchlist, param, bst, traininds, testinds

def adult_setup_newmetric(
        pct=0.8,
        baseline=True,
        nsteps=1000,
        traininds=None,
        testinds=None,
        seed=None,
        param=None,
        #dtype='float32',
        #tdtype=tf.float32,

        ):

    orig_dataset = get_adult_orig()

    X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, A, traininds, testinds = get_adult_train_test(orig_dataset, pct=pct, traininds=traininds, testinds=testinds, seed=seed)

    dtrain, dtest, watchlist, param, _, _ = adult_prep_baseline_xgb(X_train, X_test, y_train, y_test, param=param)


    # project the training data and compute pairwise distances.
    # Make the required tensorflow constant array C
    proj = script.proj_matrix_gen(X_train, A[:-1].T, False, False)
    projData = np.matmul(X_train, proj)

    print("Calculate pairwise distances... ", end=" ", flush=True)
    Corig = sp.spatial.distance.squareform(
            sp.spatial.distance.pdist(
                projData,
                metric='sqeuclidean'
            ))#.astype(dtype)
    print("done.", flush=True)

    #C = tf.constant(Corig, dtype=tdtype)

    bst=None
    if baseline:
        bst = xgb.train(param, dtrain, nsteps, watchlist)

    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, Corig, dtrain, dtest, watchlist, param, bst, traininds, testinds

def spouse_cons(bst, X_test, feature_names):

    # be super general cause we are really messing with all of
    # the feature_names and different data sets
    rel_idx = []
    husbInd = None
    wifeInd = None
    for i,name in enumerate(feature_names):
        if 'relationship' in name:
            rel_idx.append(i)
        if 'Husband' in name:
            husbInd = i
        if 'Wife' in name:
            wifeInd = i

    # Go backwards since we sometimes delete earlier features
    rel_idx = np.array(rel_idx) - len(feature_names)
    husbInd = husbInd - len(feature_names)
    wifeInd = wifeInd - len(feature_names)

    print("Comparing " + feature_names[wifeInd] + " with " +\
            feature_names[husbInd])

    # Only make one copy for SPEED
    X = np.copy(X_test)
    X[:, rel_idx] = np.zeros((X.shape[0], rel_idx.shape[0]))
    X[:, wifeInd] = np.ones(X.shape[0])
    
    predsw = bst.predict(xgb.DMatrix(X))
    y_wife = np.array([ 1 if pval > 0.5 else 0 for pval in predsw])

    X[:, wifeInd] = np.zeros(X.shape[0])
    X[:, husbInd] = np.ones(X.shape[0])

    predsh = bst.predict(xgb.DMatrix(X))
    y_husb = np.array([ 1 if pval > 0.5 else 0 for pval in predsh])

    return (y_wife*y_husb).sum() + ((1-y_wife)*(1-y_husb)).sum()

def sex_race_cons(bst, X_test, sind, rind):

    # Only make one copy for SPEED
    X = deepcopy(X_test)

    X[:, sind] = np.zeros(X.shape[0])
    X[:, rind] = np.zeros(X.shape[0])
    preds00 = bst.predict(xgb.DMatrix(X))
    y00 = np.array([ 1 if pval > 0.5 else 0 for pval in preds00])

    X[:, sind] = np.zeros(X.shape[0])
    X[:, rind] = np.ones(X.shape[0])
    preds01 = bst.predict(xgb.DMatrix(X))
    y01 = np.array([ 1 if pval > 0.5 else 0 for pval in preds01])

    X[:, sind] = np.ones(X.shape[0])
    X[:, rind] = np.zeros(X.shape[0])
    preds10 = bst.predict(xgb.DMatrix(X))
    y10 = np.array([ 1 if pval > 0.5 else 0 for pval in preds10])

    X[:, sind] = np.ones(X.shape[0])
    X[:, rind] = np.ones(X.shape[0])
    preds11 = bst.predict(xgb.DMatrix(X))
    y11 = np.array([ 1 if pval > 0.5 else 0 for pval in preds11])

    # Want all quadruples to be consistent (all 0s or all 1s)
    return (y00*y01*y10*y11).sum() + ((1-y00)*(1-y01)*(1-y10)*(1-y11)).sum()
