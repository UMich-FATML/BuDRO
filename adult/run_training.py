from optparse import OptionParser
from collections import OrderedDict
import os
import numpy as np
import tensorflow as tf
import sys

import time

import adult_proc
import script
import tf_xgboost_log as txl

_baseline_steps = 200
_baseline = False
_dtype = tf.float32

_lowpimem=True
_verbose=False

def parse_args():
    
    parser = OptionParser()

    parser.add_option("--sgd", action="store_true", dest="sgd", default=False)
    parser.add_option("--dual", action="store_true", dest="dual", default=False)

    parser.add_option("--eps", type="float", dest="eps")
    parser.add_option("--gamma", type="float", dest="gamma_reg")

    parser.add_option("--lambda", type="float", dest="lamb")
    parser.add_option("--max_depth", type="int", dest="max_depth")
    parser.add_option("--eta_lr", type="float", dest="eta_lr")
    parser.add_option("--min_child_weight", type="float", dest="min_child_weight")
    parser.add_option("--scale_pos_weight", type="float", dest="scale_pos_weight")
    parser.add_option("--n_iter", type="int", dest="n_iter")
    #parser.add_option(, type="float", dest=)
 
    parser.add_option("--sgd_init", type="float", dest="sgd_init")
    parser.add_option("--epoch", type="int", dest="epoch")
    parser.add_option("--batch_size", type="int", dest="batch_size")
    parser.add_option("--lr", type="float", dest="lr")
    parser.add_option("--momentum", type="float", dest="momentum")
    
    # seed for train/test split
    parser.add_option("--seed", type="int", dest="seed")
    
    (options, args) = parser.parse_args()
 
    return options

def main(results_path = 'YOUR_PATH_HERE'):

    options = parse_args()
    print(options)

    # method parameters
    use_sgd = options.sgd
    use_dual = options.dual

    eps = options.eps

    # xgboost parameters
    param = dict()
    param['objective'] = 'binary:logistic'
    param['max_depth'] = options.max_depth
    param['lambda'] = options.lamb
    param['eta'] = options.eta_lr
    param['min_child_weight'] = options.min_child_weight
    param['scale_pos_weight'] = options.scale_pos_weight
    n_iter = options.n_iter

    if use_sgd:
        init = options.sgd_init
        epoch = options.epoch
        batch_size = options.batch_size
        lr = options.lr
        momentum = options.momentum

        tf_eta = tf.Variable(1.0, dtype=_dtype)

    if not use_dual:
        gamma_reg = options.gamma_reg
    
    seed = options.seed
    #seed = 26869
    #seed = 18575
    
    names = ['eps', 'depth', 'eta', 'weight', 'lamb', 'pos']
    values = [ eps,
            param['max_depth'],
            param['eta'],
            param['min_child_weight'],
            param['lambda'],
            param['scale_pos_weight'],
            ]

    if not use_dual:
        names += ['gamma_reg']
        values += [gamma_reg]

    if use_sgd:
        names += ['init', 'epoch', 'momentum', 'lr']
        values += [init, epoch, momentum, lr]

    exp_descriptor = []
    for n, v in zip(names, values):
        exp_descriptor.append(':'.join([n,str(v)]))
        
    # TRAIN WITH THESE PARAMETERS
    print("LOADING ADULT DATA", flush=True)

    # This automatically takes care of scale_pos_weight in param
    X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, Corig, dtrain, dtest, watchlist, param, bst, traininds, testinds  = adult_proc.adult_setup(nsteps=200, baseline=_baseline, seed=seed,param=param, pct=0.22114, loadData=True, fileName='adult-seeds/sensr_data')

    n = Corig.shape[0]
    sind = feature_names.index('sex')
    rind = feature_names.index('race')

    # Fix the other parameter value
    param['min_child_weight'] /= n 

    # Refine the labels
    y = y_train
    yt = y_test

    inds0 = np.where(y==0)[0]
    inds1 = np.where(y==1)[0]

    C = tf.constant(Corig, _dtype)

    if _baseline:

        # find the baseline directory, if it exists
        # Create it if it doesn't exist.
        exp_folder = results_path + '_'.join(exp_descriptor)
        
        try:
            os.makedirs(exp_folder)
        except:
            pass

        ## PRINT SOME STARTING INFORMATION
        print("XGBoost parameters:")
        print(param, flush=True)

        print("BASELINE CLASSIFIER")
        p0, p1, _ = script.balanced_accuracy(bst, dtest, yt)
        print("p0: {}; p1: {}; balanced: {}".format(p0, p1, (p0+p1)/2))

        # Fairness for baseline
        predst = bst.predict(dtest)
        y_guess = np.array([1 if pval > 0.5 else 0 for pval in predst])
        print("SEX")
        base_sex = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
        print("RACE")
        base_race = script.group_metrics(yt, y_guess, y_race_test, label_good=1)

        # Save baseline information
        with open(exp_folder + '/baseline_%d' % seed, 'w') as f:
            p0, p1, _ = script.balanced_accuracy(bst, dtest, yt)
            p0t, p1t, _ = script.balanced_accuracy(bst, dtrain, y)

            scons = adult_proc.spouse_cons(bst, X_test, feature_names)
            grcons = adult_proc.sex_race_cons(bst, X_test, sind, rind)

            preds = bst.predict(dtest)
            y_guess = np.array([1 if pval > 0.5 else 0 for pval in preds])
            sex_res = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
            race_res = script.group_metrics(yt, y_guess, y_race_test, label_good=1)

            f.write('train-balanced\t{}\n'.format( (p0t+p1t)/2 ))
            f.write('test-balanced\t{}\n'.format( (p0+p1)/2 ))
            f.write('p0\t{}\n'.format(p0))
            f.write('p1\t{}\n'.format(p1))

            f.write('spouse-cons\t{}\n'.format(scons/X_test.shape[0]))
            f.write('sex-race-cons\t{}\n'.format(grcons/X_test.shape[0]))

            f.write('sex-TPR-prot\t{}\n'.format(sex_res[0]))
            f.write('sex-TNR-prot\t{}\n'.format(sex_res[1]))
            f.write('sex-TPR-priv\t{}\n'.format(sex_res[2]))
            f.write('sex-TNR-priv\t{}\n'.format(sex_res[3]))
            f.write('sex-gap-RMS\t{}\n'.format(sex_res[4]))
            f.write('sex-gap-MAX\t{}\n'.format(sex_res[5]))
            f.write('sex-ave-odds-diff\t{}\n'.format(sex_res[6]))
            f.write('sex-eq-opp-diff\t{}\n'.format(sex_res[7]))
            f.write('sex-stat-parity\t{}\n'.format(sex_res[8]))

            f.write('race-TPR-prot\t{}\n'.format(race_res[0]))
            f.write('race-TNR-prot\t{}\n'.format(race_res[1]))
            f.write('race-TPR-priv\t{}\n'.format(race_res[2]))
            f.write('race-TNR-priv\t{}\n'.format(race_res[3]))
            f.write('race-gap-RMS\t{}\n'.format(race_res[4]))
            f.write('race-gap-MAX\t{}\n'.format(race_res[5]))
            f.write('race-ave-odds-diff\t{}\n'.format(race_res[6]))
            f.write('race-eq-opp-diff\t{}\n'.format(race_res[7]))
            f.write('race-stat-parity\t{}\n'.format(race_res[8]))

    ### END BASELINE CALCULATIONS

    # Use tensorboard to examine information
    fairfile = results_path + 'tensorboard/sgd/' + '_'.join(exp_descriptor) +\
            '_result:%d' % seed
    summary_writer = tf.summary.create_file_writer(fairfile)
    #measString = "#Iter\tp0\tp1\tS-cons\t" +\
    #        "Sex-TPR-prot\tSex-TNR-prot\tSex-TPR-priv\t" +\
    #        "Sex-TNR-priv\tSex-gap-RMS\tSex-gap-max\tSex-aod\t" +\
    #        "Sex-eod\tSex-parity\t" +\
    #        "Race-TPR-prot\tRace-TNR-prot\tRace-TPR-priv\t" +\
    #        "Race-TNR-priv\tRace-gap-RMS\tRace-gap-max\tRace-aod\t" +\
    #        "Race-eod\tRace-parity\n"
    #f = open(fairfile, 'w')
    #f.write(measString)
    #f.close()

    # Define output metric here so that we can hard code
    # some of the variables (instead of passing a bunch every time we print)
    #
    # not sure why I have dtrain and dtest as inputs... but whatever
    def tf_metric(bst, dtrain, dtest, max_loss, it, slack=None):
        p0, p1, _ = script.balanced_accuracy(bst, dtest, yt)
        p0t, p1t, _ = script.balanced_accuracy(bst, dtrain, y)

        scons = adult_proc.spouse_cons(bst, X_test, feature_names)
        grcons = adult_proc.sex_race_cons(bst, X_test, sind, rind)

        preds = bst.predict(dtest)
        y_guess = np.array([1 if pval > 0.5 else 0 for pval in preds])
        sex_res = script.group_metrics(yt, y_guess, y_sex_test, label_good=1, verbose=False)
        race_res = script.group_metrics(yt, y_guess, y_race_test, label_good=1, verbose=False)

        with summary_writer.as_default():
            tf.summary.scalar('train balanced', (p0t+p1t)/2, step=it)
            tf.summary.scalar('inner training loss', max_loss, step=it)
            tf.summary.scalar('test balanced', (p0+p1)/2, step=it)
            tf.summary.scalar('p0', p0, step=it)
            tf.summary.scalar('p1', p1, step=it)

            tf.summary.scalar('spouse cons', scons/X_test.shape[0], step=it)
            tf.summary.scalar('gr cons', grcons/X_test.shape[0], step=it)
            
            tf.summary.scalar('sex-TPR-prot', sex_res[0], step=it)
            tf.summary.scalar('sex-TNR-prot', sex_res[1], step=it)
            tf.summary.scalar('sex-TPR-priv', sex_res[2], step=it)
            tf.summary.scalar('sex-TNR-priv', sex_res[3], step=it)
            tf.summary.scalar('sex-gap-RMS', sex_res[4], step=it)
            tf.summary.scalar('sex-gap-MAX', sex_res[5], step=it)
            tf.summary.scalar('sex-ave-odds-diff', sex_res[6], step=it)
            tf.summary.scalar('sex-eq-opp-diff', sex_res[7], step=it)
            tf.summary.scalar('sex-stat-parity', sex_res[8], step=it)

            tf.summary.scalar('race-TPR-prot', race_res[0], step=it)
            tf.summary.scalar('race-TNR-prot', race_res[1], step=it)
            tf.summary.scalar('race-TPR-priv', race_res[2], step=it)
            tf.summary.scalar('race-TNR-priv', race_res[3], step=it)
            tf.summary.scalar('race-gap-RMS', race_res[4], step=it)
            tf.summary.scalar('race-gap-MAX', race_res[5], step=it)
            tf.summary.scalar('race-ave-odds-diff', race_res[6], step=it)
            tf.summary.scalar('race-eq-opp-diff', race_res[7], step=it)
            tf.summary.scalar('race-stat-parity', race_res[8], step=it)

            if use_sgd:
                tf.summary.scalar('slack', slack, step=it)

        summary_writer.flush()

        #res = str(p0) + "\t" + str(p1) + "\t" +\
        #        "\t".join(str(x) for x in sex_res) + "\t" +\
        #        "\t".join(str(x) for x in race_res) + "\n"

        #return res
    
    ## FAIR TRAINING
    t_s = time.time()
    if use_dual and use_sgd:
        fst, xport = txl.boost_dual_sgd(
                X_train,
                y,
                Corig,
                C,
                n_iter, 
                tf_eta,
                X_test=X_test,
                y_test=yt,
                pred=None, 
                eps=eps, 
                epoch=epoch,
                batch_size=batch_size,
                lr=lr,
                momentum=momentum,
                init=init,
                param=param,
                verbose=False,
                verify=True,
                dtype=_dtype,
                outfunc=tf_metric,
                outinterval=1,
                lowpimem=_lowpimem,
                )

        stuff = None

    elif use_dual and not use_sgd:
        fst, xport, stuff =  txl.boost_dual(
                X_train,
                y,
                C, 
                n_iter, 
                X_test=X_test,
                y_test=yt,
                pred=None, 
                eps=eps, 
                param=param,
                verbose=False,
                verify=True,
                lowg=0.0,
                highg=10.0,
                method='brentq',
                dtype=_dtype,
                outfunc=tf_metric,
                )
    elif not use_dual and not use_sgd:
        fst, pi = txl.boost_sinkhorn_tf(
                X_train,
                y,
                C,
                n_iter, 
                X_test=X_test,
                y_test=yt,
                pred=None, 
                eps=eps, 
                gamma_reg=gamma_reg,
                param=param,
                lowg=0.0,
                highg=5.0,
                verbose=False,
                verify=True,
                roottol=10**(-16),
                dtype=_dtype,
                max_float_pow = 65.,
                outfunc=tf_metric
                )

    
    print('Took %f' % (time.time() - t_s))

    # output some summary imformation
    if _verbose:
        print("BOOSTED CLASSIFIER FINAL INFORMATION")
        p0, p1, _ = script.balanced_accuracy(fst, dtest, yt)
        print("p0: {}; p1: {}; balanced: {}".format(p0, p1, (p0+p1)/2))

        predst = fst.predict(dtest)
        y_guess = np.array([1 if pval > 0.5 else 0 for pval in predst])
        print("SEX")
        script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
        print("RACE")
        script.group_metrics(yt, y_guess, y_race_test, label_good=1)

    final = 0
    if use_dual:
        dists = Corig[xport, np.arange(Corig.shape[1])]

        if stuff is not None:
            dists[stuff[0]] = 0
            final += Corig[stuff[0], stuff[2]]*stuff[3] +\
                    Corig[stuff[0], stuff[1]]*(1/n - stuff[3])

        final += dists.sum()/n
    else:
        final = (Corig*pi.numpy()).sum()

    print("Constraint value: {}".format(final))
    print('DONE')
    
    return 0

if __name__ == '__main__':
    main()
