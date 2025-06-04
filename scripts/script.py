import numpy as np
import scipy as sp
import xgboost as xgb

from itertools import product

# Some scripts and whatnot to help with processing

def balanced_accuracy(bst, data, y_test):

    preds = bst.predict(data)
    y_guess = np.array([1 if pred > 0.5 else 0 for pred in preds])

    inds0 = np.where(y_test == 0)[0]
    inds1 = np.where(y_test == 1)[0]

    num0 = (1-y_test).sum()
    num1 = y_test.sum()

    p0 = (num0 - y_guess[inds0].sum())/num0
    p1 = y_guess[inds1].sum()/num1

    return p0, p1, [num0, num1, inds0, inds1]

def gen_balanced_accuracy(y_test, y_guess=None, preds=None):

    if y_guess is None and preds is None:
        print("Error: need to input guess or pred values")
        return

    if y_guess is None:
        y_guess = np.array([1 if pred > 0.5 else 0 for pred in preds])

    inds0 = np.where(y_test == 0)[0]
    inds1 = np.where(y_test == 1)[0]

    num0 = (1-y_test).sum()
    num1 = y_test.sum()

    p0 = (num0 - y_guess[inds0].sum())/num0
    p1 = y_guess[inds1].sum()/num1

    return p0, p1, [num0, num1, inds0, inds1]

def proj_matrix_gen(X_train, A, test=False, save=False):

    ProjA = np.matmul(
        np.matmul( A, np.linalg.inv(np.matmul(A.T, A))),
        A.T
    )

    print("Shape of projection matrix: {}".format(ProjA.shape))

    if test:
        print("Projection error: {}".format(
            np.abs(np.matmul(ProjA, ProjA) - ProjA).max()
        ))

    dmat = np.eye(X_train.shape[1]) - ProjA
    if save: np.savez("adult-mat.npz", mat=dmat)

    return dmat

def group_metrics(y_true, y_pred, y_protected, label_protected=0, label_good=0, verbose=True):
    # index of priv and protected
    idx_prot = np.where(y_protected == label_protected)[0]
    idx_priv = np.where(y_protected != label_protected)[0]

    # good = positive outcome, bad = negative outcome
    idx_good_class = np.where(y_true == label_good)[0]
    idx_pred_good_class = np.where(y_pred == label_good)[0]
    idx_bad_class = np.where(y_true != label_good)[0]
    idx_pred_bad_class = np.where(y_pred != label_good)[0]

    correct = y_true==y_pred
	
    TPR_prot = correct[np.intersect1d(idx_good_class, idx_prot)].mean()
    FP_prot = (1-correct[np.intersect1d(idx_pred_good_class, idx_prot)]).sum()
    FPR_prot = FP_prot/len(np.intersect1d(idx_bad_class, idx_prot))
    TPR_priv = correct[np.intersect1d(idx_good_class, idx_priv)].mean()
    FP_priv = (1-correct[np.intersect1d(idx_pred_good_class, idx_priv)]).sum()
    FPR_priv = FP_priv/len(np.intersect1d(idx_bad_class, idx_priv))

    gap_TPR = TPR_prot - TPR_priv
    gap_TNR = (1-FPR_prot) - (1-FPR_priv)

    gap_rms = np.sqrt( 0.5 * (gap_TPR**2 + gap_TNR**2))
    gap_max = np.max((np.abs(gap_TPR), np.abs(gap_TNR)))
    
    accuracy = sum(y_true[_] == y_pred[_] for _ in range(len(y_true))) / len(y_pred)

    average_odds_difference = ((TPR_prot - TPR_priv) + (FPR_prot - FPR_priv))/2
    equal_opportunity_difference = TPR_prot - TPR_priv
    statistical_parity_difference = (y_pred[idx_prot]==label_good).mean() - (y_pred[idx_priv]==label_good).mean()

    if verbose:
        print('Accuracy is %f' % accuracy)

        print("Protected class:")
        print("TPR: {}; TNR: {}".format(TPR_prot, 1-FPR_prot))
        print("Priv class:")
        print("TPR: {}; TNR: {}".format(TPR_priv, 1-FPR_priv))

        print('Gap RMS: {}, Gap MAX: {}'.format(gap_rms, gap_max))
    
        print('Average odds difference is %f' % average_odds_difference)
        print('Equal opportunity difference is %f' % equal_opportunity_difference)
        print('Statistical parity difference is %f' % statistical_parity_difference)
    
    return TPR_prot, 1-FPR_prot, TPR_priv, 1-FPR_priv, gap_rms, gap_max, average_odds_difference, equal_opportunity_difference, statistical_parity_difference


# Assuming that all labels and class indicators are 0 vs 1
# This appears to be working in my simple test examples.
def class_weights(protected, labels):

    class_labels = [0,1]
    classes = [0,1]

    num = protected.shape[0]

    num_protected = 1
    if len(protected.shape) > 1:
        num_protected = protected.shape[1]


    pos_counts = np.zeros(2**num_protected)
    neg_counts = np.zeros(2**num_protected)

    for i, row in enumerate(protected):
        ind = 0
        for elt in row:
            ind = ind << 1
            if elt > class_labels[0]: ind += 1

        if labels[i]: pos_counts[ind] += 1
        else: neg_counts[ind] += 1
        
    counts = pos_counts + neg_counts
    pos = pos_counts.sum()
    neg = neg_counts.sum()

    weights = np.zeros((2**num_protected, 2))

    for ind in range(weights.shape[0]):
        weights[ind][0] = counts[ind] * neg / (num * neg_counts[ind])
        weights[ind][1] = counts[ind] * pos / (num * pos_counts[ind])

    return weights, (pos_counts, neg_counts)

