import sys

# path to fair boosting scripts
sys.path.append('../scripts')

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

import german_proc
import SenSR
# the file SenSR.py from github.com/IBM/sensitive-subspace-robustness
import script

seeds = np.load("german-seeds.npz")['seeds']

# This code copied and edited from SenSR as much as possible
def get_consistency(X, weights=0, proj = 0, status_inds=[37,38,39,40], adv = 0, dataset_orig_test = 0):

    # keeping this the same... just run it
    if adv == 0:
        N, D = X.shape
        K = 1

        tf_X = tf.placeholder(tf.float32, shape=[None,D])
        tf_y = tf.placeholder(tf.float32, shape=[None,K], name='response')

        n_units = weights[1].shape
        n_units = n_units[0]

        _, l_pred, _, _ = SenSR.forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = tf.nn.relu)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n, _ = X.shape

        # make 4 versions of the original data by changing binary gender and gender, then count how many classifications change
        #copy 1
        X00 = np.copy(X)
        X00[:, status_inds] = 0
        X00[:, status_inds[0]] = 1

        if np.ndim(proj) != 0:
            X00 = X00@proj

        if adv == 0:
            X00_logits = l_pred.eval(feed_dict={tf_X: X00})
            X00_preds = np.argmax(X00_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X00
            dataset_mod, _ = adv.predict(dataset_mod)
            X00_preds = [i[0] for i in dataset_mod.labels]

        #### copy 2
        X01 = np.copy(X)
        X01[:, status_inds] = 0
        X01[:, status_inds[1]] = 1

        if np.ndim(proj) != 0:
            X01 = X01@proj

        if adv == 0:
            X01_logits = l_pred.eval(feed_dict={tf_X: X01})
            X01_preds = np.argmax(X01_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X01
            dataset_mod, _ = adv.predict(dataset_mod)
            X01_preds = [i[0] for i in dataset_mod.labels]

        #### copy 3
        X10 = np.copy(X)
        X10[:, status_inds] = 0
        X10[:, status_inds[2]] = 1

        if np.ndim(proj) != 0:
            X10 = X10@proj

        if adv == 0:
            X10_logits = l_pred.eval(feed_dict={tf_X: X10})
            X10_preds = np.argmax(X10_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X10
            dataset_mod, _ = adv.predict(dataset_mod)
            X10_preds = [i[0] for i in dataset_mod.labels]

        #### copy 4
        X11 = np.copy(X)
        X11[:, status_inds] = 0
        X11[:, status_inds[3]] = 1

        if np.ndim(proj) != 0:
            X11 = X11@proj

        if adv == 0:
            X11_logits = l_pred.eval(feed_dict={tf_X: X11})
            X11_preds = np.argmax(X11_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X11
            dataset_mod, _ = adv.predict(dataset_mod)
            X11_preds = [i[0] for i in dataset_mod.labels]

        status_consistency = np.mean([1 if X00_preds[i] == X01_preds[i] and X00_preds[i] == X10_preds[i] and X00_preds[i] == X11_preds[i] else 0 for i in range(len(X00_preds))])

        return status_consistency

    
# Assuming we want to ignore the binary age
def german_nns(lr=0.0001, epoch=4100, filename='german-NN-data.npz'):

    dayta = german_proc.get_german_data()

    # Collect p0, p1, balacc, age TPR/TNR, age GAPS, 3x group metrics
    # sex TPR/TNR, sex GAPS, 3x group metrics
    # Look at consistency among all four types of personal status
    finfo = np.zeros((seeds.shape[0], 21))
    consinfo = np.zeros((seeds.shape[0], 1))

    for iseed, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test, y_age_train, y_age_test, feature_names, traininds, testinds = german_proc.get_german_train_test_age(dayta, pct=0.8, removeProt=False, seed=seed)

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

        # Get the age coordinates
        y_age_bin = X_train[:,abind]
        yt_age = X_test[:, abind]
        
        # Make one_hot encoded y for train_nn
        one_hot = OneHotEncoder(sparse=False)
        one_hot.fit(y_train.reshape(-1,1))
        
        yt = one_hot.transform(y_test.reshape(-1,1))
        y = one_hot.transform(y_train.reshape(-1,1))

        weights, train_logits, test_logits = SenSR.train_nn(X_train, y, X_test=X_test, y_test=yt, n_units=[100], lr=lr, epoch=epoch, verbose=False)

        preds = np.argmax(test_logits, axis=1)

        p0,p1, _ = script.gen_balanced_accuracy(y_test, preds)

        # consistency measures...
        scons = get_consistency(X_test, weights=weights, status_inds=status_inds)
        consinfo[iseed, 0] = scons

        print("GENDER INFO")
        sex_res = script.group_metrics(
                y_test,
                preds,
                yt_sex,
                label_good=1,
                verbose=True
                )

        print("AGE INFO")
        age_res = script.group_metrics(
                y_test,
                preds,
                yt_age,
                label_good=1,
                verbose=True
                )

        finfo[iseed,0] = p0
        finfo[iseed,1] = p1
        finfo[iseed,2] = (p0 + p1)/2

        finfo[iseed, range(3,12)] = age_res
        finfo[iseed, range(12,21)] = sex_res


    np.savez(filename, finfo=finfo, consinfo=consinfo)






