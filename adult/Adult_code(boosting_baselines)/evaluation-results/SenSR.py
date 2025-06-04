import numpy as np
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from collections import OrderedDict
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# from script import get_consistency, get_metrics
import script
tf.compat.v1.disable_eager_execution()

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

def fair_dist(proj, w=0.):
    tf_proj = tf.constant(proj, dtype=tf.float32)
    if w>0:
        return lambda x, y: tf.reduce_sum(input_tensor=tf.square(tf.matmul(x-y,tf_proj)) + w*tf.square(tf.matmul(x-y,tf.eye(proj.shape[0]) - tf_proj)), axis=1)
    else:
        return lambda x, y: tf.reduce_sum(input_tensor=tf.square(tf.matmul(x-y,tf_proj)), axis=1)

def weight_variable(shape, name):
    if len(shape)>1:
        init_range = np.sqrt(6.0/(shape[-1]+shape[-2]))
    else:
        init_range = np.sqrt(6.0/(shape[0]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32) # seed=1000
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def sample_batch_idx(y, n_per_class):
    batch_idx = []
    for i in range(y.shape[1]):
        batch_idx += np.random.choice(np.where(y[:,i]==1)[0], size=n_per_class, replace=False).tolist()

    np.random.shuffle(batch_idx)
    return batch_idx

def fc_network(variables, layer_in, n_layers, l=0, activ_f = tf.nn.relu, units = []):
    if l==n_layers-1:
        layer_out = tf.matmul(layer_in, variables['weight_'+str(l)]) + variables['bias_' + str(l)]
        units.append(layer_out)
        return layer_out, units
    else:
        layer_out = activ_f(tf.matmul(layer_in, variables['weight_'+str(l)]) + variables['bias_' + str(l)])
        l += 1
        units.append(layer_out)
        return fc_network(variables, layer_out, n_layers, l=l, activ_f=activ_f, units=units)

def forward(tf_X, tf_y, weights=None, n_units = None, activ_f = tf.nn.relu, l2_reg=0.):

    if weights is not None:
        n_layers = int(len(weights)/2)
        n_units = [weights[i].shape[0] for i in range(0,len(weights),2)]
    else:
        n_features = int(tf_X.shape[1])
        n_class = int(tf_y.shape[1])
        n_layers = len(n_units) + 1
        n_units = [n_features] + n_units + [n_class]

    variables = OrderedDict()
    if weights is None:
        for l in range(n_layers):
            variables['weight_' + str(l)] = weight_variable([n_units[l],n_units[l+1]], name='weight_' + str(l))
            variables['bias_' + str(l)] = bias_variable([n_units[l+1]], name='bias_' + str(l))
    else:
        weight_ind = 0
        for l in range(n_layers):
            variables['weight_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1
            variables['bias_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1


    ## Defining NN architecture
    l_pred, units = fc_network(variables, tf_X, n_layers, activ_f = activ_f)

    cross_entropy = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(tf_y), logits=l_pred))

    correct_prediction = tf.equal(tf.argmax(input=l_pred, axis=1), tf.argmax(input=tf_y, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

    if l2_reg > 0:
        loss = cross_entropy + l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
    else:
        loss = cross_entropy

    return variables, l_pred, loss, accuracy

def train_nn(X_train, y_train, X_test=None, y_test=None, weights=None, n_units = [1000], lr=0.001, batch_size=1000, epoch=4000, verbose=True, activ_f = tf.nn.relu, l2_reg=0.):
    N, D = X_train.shape
#     print('X_train:', X_train.shape)

    try:
        K = y_train.shape[1]
#         print('y_train:', y_train.shape)
#         print('K:', K)
    except:
        K = len(weights[-1])

    tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None,D])
    tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None,K], name='response')

    variables, l_pred, loss, accuracy = forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = activ_f, l2_reg=l2_reg)

    if epoch > 0:
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        n_per_class = int(batch_size/K)
        n_per_class = int(min(n_per_class, min(y_train.sum(axis=0))))
        batch_size = int(K*n_per_class)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for it in range(epoch):
            batch_idx = sample_batch_idx(y_train, n_per_class)

            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            train_step.run(feed_dict={
                  tf_X: batch_x, tf_y: batch_y})

            if it % 10 == 0 and verbose:
                print('\nEpoch %d train accuracy %f' % (it, accuracy.eval(feed_dict={
                      tf_X: X_train, tf_y: y_train})))
                if y_test is not None:
                    print('Epoch %d test accuracy %g' % (it, accuracy.eval(feed_dict={
                          tf_X: X_test, tf_y: y_test})))
        if y_train is not None:
            print('\nFinal train accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_train, tf_y: y_train})))
        if y_test is not None:
            print('Final test accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_test, tf_y: y_test})))

        weights = [x.eval() for x in variables.values()]
        train_logits = l_pred.eval(feed_dict={tf_X: X_train})
        if X_test is not None:
            test_logits = l_pred.eval(feed_dict={tf_X: X_test})
        else:
            test_logits = None

    return weights, train_logits, test_logits

