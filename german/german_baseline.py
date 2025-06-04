from optparse import OptionParser
from itertools import product
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
import numpy as np
import xgboost as xgb

import sys
sys.path.append('../')

import german_proc


def parse_args():
    
    parser = OptionParser()

    # seed for train/test split
    parser.add_option("--seed", type="int", dest="seed")

    # project out the protected subspace?
    parser.add_option(
            "--project", action="store_true", dest="project", default=False
            )

    # possibly include a binarized age variable.  You don't want to do this.
    parser.add_option(
            "--include_age_bin", action="store_true", dest="agebin", default=False
            )

    (options, args) = parser.parse_args()
 
    return options

# Copied and pasted from SenSR github.com/IBM/sensitive-subspace-robustness
def compl_svd_projector(names, svd=-1):
    if svd > 0:
        tSVD = TruncatedSVD(n_components=svd)
        tSVD.fit(names)
        basis = tSVD.components_.T
        print('Singular values:')
        print(tSVD.singular_values_)
    else:
        basis = names.T

    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj_compl = np.eye(proj.shape[0]) - proj
    return proj_compl

def balacc(predt: np.ndarray, dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    y_guess = np.array([1 if pval >= 0.5 else 0 for pval in predt])

    yt0 = np.where(y_true==0)[0]
    yt1 = np.where(y_true==1)[0]

    p0 = (yt0.shape[0] - y_guess[yt0].sum())/yt0.shape[0]
    p1 = y_guess[yt1].sum()/yt1.shape[0]

    return 'balacc', (p0 + p1)/2

def main(results_path = 'YOUR_RESULTS_PATH_HERE'):

    options = parse_args()
    print(options)
    seed = options.seed
    project = options.project
    agebin = options.agebin

    # FIX THIS BEFORE PROJECTING
    #project=False

    # Load the correct data
    orig_dataset = german_proc.get_german_data()
    X_train, X_test, y_train, y_test, y_age_train, y_age_test, feature_names, traininds, testinds = german_proc.get_german_train_test_age(orig_dataset, pct=0.8, removeProt=False, seed=seed)
    #X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_age_train, y_age_test, feature_names, A, traininds, testinds = german_proc.get_german_train_test(orig_dataset, pct=0.8, removeProt=True, seed=seed)

    aind = feature_names.index('age')
    abind = feature_names.index('age_bin')

    if not agebin:
        print("Eliminating binary age attribute")
        X_train[:,abind] = 0
        X_test[:, abind] = 0

    if project:
        print("Projecting tha dayta!")
        RCV = german_proc.train_ridge(X_train, y_age_train, aind=aind, abind=None)
        proj = german_proc.german_proj_mat(RCV, aind)
        #proj = compl_svd_projector(A, svd=A.shape[0])
        X_train = np.matmul(X_train, proj)
        X_test = np.matmul(X_test, proj)

    dtrain, dtest, watchlist, param, _, _ = german_proc.prep_baseline_xgb(X_train, X_test, y_train, y_test)

    # xgboost parameter grid - for baseline
    lambda_grid = [0.1, 10, 100, 250, 500, 750, 1000, 
            1250, 1500, 1750, 2000, 2500] #12

    depth_grid = [4,7,10,13] #4
    eta_grid = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5] #6
    weight_grid = [0.01, 0.1, 1, 2, 5] #5

    pos_grid = [0.0]  #1
    n_iter = 1000

    #if project:
    # Use a smaller grid now that we are tuned in to where things
    # are good
    lambda_grid = [0.1, 10, 100, 250, 500, 1000, 1500, 2000] #8

    depth_grid = [4,7,10] #3
    eta_grid = [0.05, 0.1, 0.3, 0.5] #4
    weight_grid = [0.01, 0.1, 1, 2, 5] #5


    # test the script
    #lambda_grid = [ 100 ]
    #depth_grid = [4,7,10,13]
    #eta_grid = [0.05] 
    #weight_grid = [ 2] 
    #pos_grid = [0.0]
    #n_iter = 1000

    hypers = [depth_grid, eta_grid, weight_grid, lambda_grid, pos_grid]
    names = ['depth', 'eta', 'weight', 'lamb', 'pos']

    names += ['seed']

    for pack in product(*hypers):

        param['max_depth'] = pack[0]
        param['eta'] = pack[1]
        param['min_child_weight'] = pack[2]
        param['lambda'] = pack[3]
        param['scale_pos_weight'] += pack[4]

        ## PRINT SOME STARTING INFORMATION
        print("XGBoost parameters:")
        print(param, flush=True)

        res = dict()

        values = list(pack)
        values.append(seed)

        exp_descriptor = []
        for n, v in zip(names, values):
            exp_descriptor.append(':'.join([n,str(v)]))
            
        exp_name = '_'.join(exp_descriptor)
        print(exp_name)

        bst = xgb.train(
                param,
                dtrain,
                n_iter,
                feval=balacc,
                evals=watchlist,
                evals_result=res,
                verbose_eval=False
                )

        # convert result into an array and save it
        res_array = np.zeros((n_iter, 4))
        res_array[:,0] = np.array(res['train']['error'])
        res_array[:,1] = np.array(res['train']['balacc'])
        res_array[:,2] = np.array(res['test']['error'])
        res_array[:,3] = np.array(res['test']['balacc'])

        print("saving data")
        np.savetxt(results_path + exp_name, res_array)

if __name__ == '__main__':
    main()
