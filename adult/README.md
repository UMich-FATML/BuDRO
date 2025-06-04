This folder contains the code for the experiments on the Adult data set.

Essentially, `run_training.py` contains the implementation of BuDRO on Adult.
It relies on the files in the directory `../scripts` - please copy them into
the working directory if you wish to run the code below. 

See the file `submit.py` for the grid of hyperparameters that were used during
training for our experiments.  In summary: use dual sgd version of BuDRO.
Vary eps, lambda, max_depth, eta_lr, min_child_weight, and the sgd lr.
Information about timing and the required memory for our jobs is also found at
the end of `submit.py`.

To run, call:
python run_training.py

It accepts the following arguments:
--sgd   (boolean, whether or not to use SGD method; default: TRUE)
--dual  (boolean, whether or not to use dual method; default: TRUE)

--eps   (float, the perturbation budget)
--gamma (float, the entropic regulariation parameter, ignored when --dual)
--seed  (int, seed for train/test split)
--n_iter        (int, number of boosting steps) 

# XGBoost parameters
--lambda        (float, l2 regularization parameter)
--max_depth     (int, max depth of tree)
--eta_lr        (float, XGBoost learning rate )
--min_child_weight      (float, tree regularization parameter)
--scale_pos_weight      (float, additive offset 

# SGD parameters
--sgd_init      (float, sgd initialization parametere)
--epoch         (int, sgd epoch)
--batch_size    (int, sgd batch size)
--lr            (float, sgd learning rate)
--momentum      (float, sgd momentum parameter)


Running this requires that you already have the train/test split data for the
input seed in a separate file.  See below for how to generate these train/test
split data files.

The file `submit.py` uses tensorboard to write information about the run to
specific data files.  We use these files to generate Figures 2 and 3, found in
the Appendix.  The code used to generate these figures appears in the notebook
`adult-pics.ipynb`.

As discussed in the Appendix, we use Figure 2 to help select the
hyperparameters that we use for the results in the tables in the main text.
We do this by comparing several hyperparameter selections that achieve
desirable results in Figure 2 and attempt to generalize these hyperparameters
a bit (we generalize like this to simulate hand-tuning - it seems more
reasonable that we would be able to stumble upon these good hyperparameters if
they create several good datapoints in Figure 2).

The deliberations for selecting these final hyperparameters are found in the
file `adult-params-highacc.txt`.  Some of the final hyperparameters are at the
top of this file.  All hyperparameters from all points considered are found in
the file `adult-select-test-params.txt`.


## Generating train/test splits for use with Adult

The seeds that we used in our trials are included here in the binary files:
`adult-seeds.npz` and `adult-seeds-test.npz`
Both of those files include a list of integer seeds:

>>> import numpy as np
>>> dayta = np.load("adult-seeds.npz")
>>> dayta['seeds']


We use the adult.py file from SenSR (github.com/IBM/sensitive-subspace-robustness)
to generate the splits for use with adult.  This is to ensure that we are
preprocessing in exactly the same way for comparisons. 


### TO GENERATE AN ADULT TRAIN/TEST SPLIT

>>> import adult  // refers to the file adult.py from the SenSR repository 
>>> X_train, X_test, y_train, y_test, X_gender_train, X_gender_test,y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test = adult.preprocess_adult_data(seed=seed, pct=.8)

s>>> sensitive_directions, proj = adult.get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train,  X_gender_test, y_gender_test)

>>> feature_names = np.array(dataset_orig_train.feature_names)
>>> np.savez('sensr_data_' + str(seed) + '.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=feature_names, proj=proj)


### To load an adult train/test split for use with BuDRO

The following code automatically trains a baseline GBDT (using XGBoost) on the
train/test split data that is found in the file `sensr_data_SEED.npz`
This baseline GBDT uses the parameters in the dictionary `param` (defined
below).  The baseline is trained for balanced accuracy.  Feel free to tweak
the parameters below to obtain different baseline results.

```
import adult_proc

param = {
        'max_depth':5,
        'eta':0.05,
        'objective':'binary:logistic',
        'min_child_weight':0.5,
        'lambda':0.0001,
}

(
    X_train,
    X_test,
    y_train,
    y_test,
    y_sex_train,
    y_sex_test,
    y_race_train,
    y_race_test,
    feature_names,
    Corig,      # distance matrix
    dtrain,     # XGBoost object containing training data
    dtest,
    watchlist,
    param,      
    bst,        # basline GBDT object
    traininds,
    testinds
)  = adult_proc.adult_setup(
    nsteps=200, # number of boosting steps for baseline
    baseline=True, 
    seed=SEED, 
    param=param,
    pct=0.22114, # ignored when loadData = True
    loadData=True,
    fileName='sensr_data' # prefix of file name, ending with 'SEED.npz'
)
```

