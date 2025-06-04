import numpy as np
import scipy as sp
import tensorflow as tf
import xgboost as xgb

def balanced_accuracy(y_guess, y_test):

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


def group_metrics(y_true, y_pred, y_protected, label_protected=0, label_good=0):
    
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
#     print('Accuracy is %f' % accuracy)

#     print("Protected class:")
#     print("TPR: {}; TNR: {}".format(TPR_prot, 1-FPR_prot))
#     print("Priv class:")
#     print("TPR: {}; TNR: {}".format(TPR_priv, 1-FPR_priv))

#     print('Gap RMS: {}, Gap MAX: {}'.format(gap_rms, gap_max))
    
    average_odds_difference = ((TPR_prot - TPR_priv) + (FPR_prot - FPR_priv))/2
#     print('Average odds difference is %f' % average_odds_difference)
    
    equal_opportunity_difference = TPR_prot - TPR_priv
#     print('Equal opportunity difference is %f' % equal_opportunity_difference)
    
    statistical_parity_difference = (y_pred[idx_prot]==label_good).mean() - (y_pred[idx_priv]==label_good).mean()
#     print('Statistical parity difference is %f' % statistical_parity_difference)
    
    return TPR_prot, 1-FPR_prot, TPR_priv, 1-FPR_priv, gap_rms, gap_max, average_odds_difference, equal_opportunity_difference, statistical_parity_difference, accuracy

