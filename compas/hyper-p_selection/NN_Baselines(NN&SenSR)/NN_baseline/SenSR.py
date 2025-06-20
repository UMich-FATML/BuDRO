import numpy as np
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from collections import OrderedDict
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# from script import get_consistency, get_metrics
import script
# tf.compat.v1.disable_eager_execution()

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

def train_nn(row_list, dataset_orig_test, X_train, y_train, X_test=None, y_test=None, weights=None, n_units = [1000], lr=0.001, batch_size=1000, epoch=4000, verbose=True, activ_f = tf.nn.relu, l2_reg=0.):
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

            if it % 50 == 0 and it >= 1000:
                print('\nEpoch %d train accuracy %f' % (it, accuracy.eval(feed_dict={
                      tf_X: X_train, tf_y: y_train})))
                if y_test is not None:
                    print('Epoch %d test accuracy %g' % (it, accuracy.eval(feed_dict={
                          tf_X: X_test, tf_y: y_test})))
#                                 print('X_test shape:', X_test.shape)
                    test_logits = l_pred.eval(feed_dict={tf_X: X_test})
                    preds = np.argmax(test_logits, axis = 1)
                    y_guess = preds

                    y_t = y_test[:,1]
#                     print('preds shape:', preds.shape)
                    inds0 = np.where(y_t == 0)[0]
                    inds1 = np.where(y_t == 1)[0]

                    num0 = (1-y_t).sum()
                    num1 = y_t.sum()

                    p0 = (num0 - y_guess[inds0].sum())/num0
                    p1 = y_guess[inds1].sum()/num1
                    
                    bl_acc = (p0+p1) / 2
                    
                    acc = accuracy.eval(feed_dict={tf_X: X_test, tf_y: y_test})
                    acc1 = (y_t == y_guess).sum() / len(y_guess)
#                     print('Epoch %d test accuracy1 %g' % (it, acc1))

                    print('Saving results ......')
                    save_data = {}
                    save_data['acc'] = acc
#                     save_data['acc1'] = acc1
                    save_data['bl_acc'] = bl_acc
                    save_data['epoch'] = it
                    save_data['lr'] = lr
                    row_list.append(save_data)
                    
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

    return weights, train_logits, test_logits, row_list

def forward_fair(tf_X, tf_y, tf_fair_X, weights=None, n_units = None, activ_f = tf.nn.relu, l2_reg=0.):

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
    l_pred_fair, units_fair = fc_network(variables, tf_fair_X, n_layers, activ_f = activ_f)

    cross_entropy = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(tf_y), logits=l_pred))
    cross_entropy_fair = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(tf_y), logits=l_pred_fair))

    correct_prediction = tf.equal(tf.argmax(input=l_pred, axis=1), tf.argmax(input=tf_y, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

    if l2_reg > 0:
        cross_entropy += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
        cross_entropy_fair += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])

    return variables, l_pred, cross_entropy, accuracy, cross_entropy_fair


def compute_gap_RMS_and_gap_max(data_set):
    '''
    Description: computes the gap RMS and max gap
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = -1*data_set.false_negative_rate_difference()
    TNR = -1*data_set.false_positive_rate_difference()

    return np.sqrt(1/2*(TPR**2 + TNR**2)), max(np.abs(TPR), np.abs(TNR))

def compute_balanced_accuracy(data_set):
    '''
    Description: computes the balanced accuracy, i.e. the average of TPR and TNR
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = data_set.true_positive_rate()
    TNR = data_set.true_negative_rate()
    return 0.5*(TPR+TNR)


def get_consistency(X, weights=0):
    '''
    Description: Ths function computes spouse consistency and gender and race consistency.
    Input:
        X: numpy matrix of predictive features
        weights: learned weights for project, baseline, and sensr
        proj: if using the project first baseline, this is the projection matrix
        gender_idx: column corresponding to the binary gender variable
        race_idx: column corresponding to the binary race variable
        relationship)_idx: list of column for the following features: relationship_ Husband, relationship_ Not-in-family, relationship_ Other-relative, relationship_ Own-child, relationship_ Unmarried, relationship_ Wife
        husband_idx: column corresponding to the husband variable
        wife_idx: column corresponding to the wife variable
        adv: the adversarial debiasing object if using adversarial Adversarial Debiasing
        dataset_orig_test: this is the data in a BinaryLabelDataset format when using adversarial debiasing
    '''
#     gender_race_idx = [gender_idx, race_idx]
    gender_idx = 0
    race_idx = 1
    N, D = X.shape
    K = 1

    tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None,D])
    tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None,K], name='response')

    n_units = weights[1].shape
    n_units = n_units[0]

    _, l_pred, _, _ = forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = tf.nn.relu)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        n, _ = X.shape

        # make 4 versions of the original data by changing binary gender and gender, then count how many classifications change
        # copy 1
        X0 = np.copy(X)
        X0[:, race_idx] = 0

        X0_logits = l_pred.eval(feed_dict={tf_X: X0})
        X0_preds = np.argmax(X0_logits, axis = 1)

        ## copy 2
        X1 = np.copy(X)
        X1[:, race_idx] = 1

        X1_logits = l_pred.eval(feed_dict={tf_X: X1})
        X1_preds = np.argmax(X1_logits, axis = 1)
        
        race_consistency =  (X0_preds*X1_preds).sum() + ((1-X0_preds)*(1-X1_preds)).sum()
        race_consistency /= n

        ### copy 3
        Xg0 = np.copy(X)
        Xg0[:, gender_idx] = 0

        Xg0_logits = l_pred.eval(feed_dict={tf_X: Xg0})
        Xg0_preds = np.argmax(Xg0_logits, axis = 1)

        #### copy 4
        Xg1 = np.copy(X)
        Xg1[:, gender_idx] = 1

        Xg1_logits = l_pred.eval(feed_dict={tf_X: Xg1})
        Xg1_preds = np.argmax(Xg1_logits, axis = 1)
        
        gender_consistency =  (Xg0_preds*Xg1_preds).sum() + ((1-Xg0_preds)*(1-Xg1_preds)).sum()
        gender_consistency /= n

        return race_consistency, gender_consistency
    
    
def get_metrics(dataset_orig, preds):
    '''
    Description: This code computes accuracy, balanced accuracy, max gap and gap rms for race and gender
    Input: dataset_orig: a BinaryLabelDataset (from the aif360 module)
            preds: predictions
    '''
    dataset_learned_model = dataset_orig.copy()
    dataset_learned_model.labels = preds

    # wrt gender
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    classified_metric = ClassificationMetric(dataset_orig,
                                                     dataset_learned_model,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    bal_acc = compute_balanced_accuracy(classified_metric)

    return bal_acc


def train_fair_nn(seed, row_list, dataset_orig_test, sind, rind, X_train, y_train, sensitive_directions, X_test=None, y_test=None, weights=None, n_units = [1000], lr=0.001, batch_size=1000, epoch=4000, verbose=False, activ_f = tf.nn.relu, l2_reg=0., lamb_init=2., subspace_epoch=10, subspace_step=0.1, eps=None, full_step=-1, full_epoch=10, fair_start = 0):

    ## Fair distance
    proj_compl = compl_svd_projector(sensitive_directions, svd=-1)
    dist_f = fair_dist(proj_compl, 0.)
    V_sensitive = sensitive_directions.shape[0]

#     global_step = tf.contrib.framework.get_or_create_global_step()

    N, D = X_train.shape
    lamb = lamb_init
    
    y_sex_test = X_test[:, sind]
    y_race_test = X_test[:, rind]

    try:
        K = y_train.shape[1]
    except:
        K = len(weights[-1])

    n_per_class = int(batch_size/K)
    n_per_class = int(min(n_per_class, min(y_train.sum(axis=0))))
    batch_size = int(K*n_per_class)

    tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None,D])
    tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None,K], name='response')

    ## Fair variables
    tf_directions = tf.constant(sensitive_directions, dtype=tf.float32)
    adv_weights = tf.Variable(tf.zeros([batch_size,V_sensitive]))
    full_adv_weights = tf.Variable(tf.zeros([batch_size,D]))
    tf_fair_X = tf_X + tf.matmul(adv_weights, tf_directions) + full_adv_weights

    variables, l_pred, _, accuracy, loss = forward_fair(tf_X, tf_y, tf_fair_X, weights=weights, n_units = n_units, activ_f = activ_f, l2_reg=l2_reg)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
#     train_step = optimizer.minimize(loss, var_list=list(variables.values()), global_step=global_step)
    train_step = optimizer.minimize(loss, var_list=list(variables.values()))
    reset_optimizer = tf.compat.v1.variables_initializer(optimizer.variables())
    reset_main_step = True

    ## Attack is subspace
    fair_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=subspace_step)
#     fair_step = fair_optimizer.minimize(-loss, var_list=[adv_weights], global_step=global_step)
    fair_step = fair_optimizer.minimize(-loss, var_list=[adv_weights])
    reset_fair_optimizer = tf.compat.v1.variables_initializer(fair_optimizer.variables())
    reset_adv_weights = adv_weights.assign(tf.zeros([batch_size,V_sensitive]))

    ## Attack out of subspace
    distance = dist_f(tf_X, tf_fair_X)
    tf_lamb = tf.compat.v1.placeholder(tf.float32, shape=())
    dist_loss = tf.reduce_mean(input_tensor=distance)
    fair_loss = loss - tf_lamb*dist_loss

    if full_step > 0:
        full_fair_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=full_step)
        full_fair_step = full_fair_optimizer.minimize(-fair_loss, var_list=[full_adv_weights])
#         full_fair_step = full_fair_optimizer.minimize(-fair_loss, var_list=[full_adv_weights], global_step=global_step)
        reset_full_fair_optimizer = tf.compat.v1.variables_initializer(full_fair_optimizer.variables())
        reset_full_adv_weights = full_adv_weights.assign(tf.zeros([batch_size,D]))

    ######################

    failed_attack_count = 0
    failed_full_attack = 0
    failed_subspace_attack = 0

    out_freq = 50

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for it in range(epoch):

            batch_idx = sample_batch_idx(y_train, n_per_class)
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            if it > fair_start:
                if reset_main_step:
                    sess.run(reset_optimizer)
                    reset_main_step = False

                loss_before_subspace_attack = loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y})

                ## Do subspace attack
                for adv_it in range(subspace_epoch):
                    fair_step.run(feed_dict={
                            tf_X: batch_x, tf_y: batch_y})
                ## Check result
                loss_after_subspace_attack = loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y})
                if loss_after_subspace_attack < loss_before_subspace_attack:
                        sess.run(reset_adv_weights)
                        failed_subspace_attack += 1

                if full_step > 0:
                    fair_loss_before_l2_attack = fair_loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb})

                    ## Do full attack
                    for full_adv_it in range(full_epoch):
                        full_fair_step.run(feed_dict={
                            tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb})

                    ## Check result
                    fair_loss_after_l2_attack = fair_loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb})
                    if fair_loss_after_l2_attack < fair_loss_before_l2_attack:
                        sess.run(reset_full_adv_weights)
                        failed_full_attack += 1

                adv_batch = tf_fair_X.eval(feed_dict={tf_X: batch_x})

                if np.isnan(adv_batch.sum()):
                    print('Nans in adv_batch; making no change')
                    sess.run(reset_adv_weights)
                    if full_step > 0:
                        sess.run(reset_full_adv_weights)
                    failed_attack_count += 1

                elif eps is not None:
                    mean_dist = dist_loss.eval(feed_dict={tf_X: batch_x})
                    lamb = max(0.00001,lamb + (max(mean_dist,eps)/min(mean_dist,eps))*(mean_dist - eps))
            else:
                adv_batch = batch_x

            _, loss_at_update = sess.run([train_step,loss], feed_dict={
                  tf_X: batch_x, tf_y: batch_y})

            if it > fair_start:
                sess.run(reset_adv_weights)
                sess.run(reset_fair_optimizer)
                if full_step > 0:
                    sess.run(reset_full_fair_optimizer)
                    sess.run(reset_full_adv_weights)

            if it % out_freq == 0 and it>=500:
                train_acc, train_logits = sess.run([accuracy,l_pred], feed_dict={
                      tf_X: X_train, tf_y: y_train})
                print('Epoch %d train accuracy %f; lambda is %f' % (it, train_acc, lamb))
                if y_test is not None:
                    test_acc, test_logits = sess.run([accuracy,l_pred], feed_dict={
                            tf_X: X_test, tf_y: y_test})
                    print('Epoch %d test accuracy %g' % (it, test_acc))
                    
                fair_weights = [x.eval() for x in variables.values()]
                train_logits = l_pred.eval(feed_dict={tf_X: X_train})
                if X_test is not None:
                    test_logits = l_pred.eval(feed_dict={tf_X: X_test})
                else:
                    test_logits = None
                race_consistency, gender_consistency = get_consistency(X_test, weights = fair_weights)
    #             print('race_consistency', race_consistency)
    #             print('gender_consistency', gender_consistency)

                preds = np.argmax(test_logits, axis = 1)
                bl_acc = get_metrics(dataset_orig_test, preds)

                yt = dataset_orig_test.labels
                yt = np.reshape(yt, (-1, ))
                y_guess = preds
    #             print("SEX")
                base_sex = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
    #             print("RACE")
                base_race = script.group_metrics(yt, y_guess, y_race_test, label_good=1)

                print('Saving results ......')
                save_data = {}
                save_data['param'] = 'iter:%d' %it+'eps:%.3f' %eps +'se:%.6f' %subspace_epoch +'slr:%.6f' %subspace_step + 'fe:%d' %full_epoch +'flr:%.6f' %full_step
                save_data['seed'] =  seed
                save_data['acc'] = base_sex[-1]
                save_data['bl_acc'] = bl_acc
                save_data['gcons'] = gender_consistency
                save_data['rcons'] = race_consistency
                save_data['RMS(G)'] = base_sex[4]
                save_data['MAX(G)'] = base_sex[5]
                save_data['AOD(G)'] = base_sex[6]
                save_data['EOD(G)'] = base_sex[7]
                save_data['SPD(G)'] = base_sex[8]
                save_data['RMS(R)'] = base_race[4]
                save_data['MAX(R)'] = base_race[5]
                save_data['AOD(R)'] = base_race[6]
                save_data['EOD(R)'] = base_race[7]
                save_data['SPD(R)'] = base_race[8]
                row_list.append(save_data)

            if y_train is not None:
                print('\nFinal train accuracy %g' % (accuracy.eval(feed_dict={
                      tf_X: X_train, tf_y: y_train})))
            if y_test is not None:
                print('Final test accuracy %g' % (accuracy.eval(feed_dict={
                      tf_X: X_test, tf_y: y_test})))
            if eps is not None:
                print('Final lambda %f' % lamb)

    return row_list, fair_weights, train_logits, test_logits
