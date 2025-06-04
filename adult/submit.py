from optparse import OptionParser
import os
import numpy as np
from itertools import product
from subprocess import call

# Submit jobs for training on adult
# Actually don't need any GPUs for training
# One trial will take ~1.5 hours on one CPU with 30GB mem
#
# Replace parameters with optimal parameters
# and replace seeds with test-seeds
# to reproduce test data from paper

def parse_args():
    
    parser = OptionParser()
    parser.set_defaults()
    
    parser.add_option("--sgd", action="store_true", dest="sgd", default=False)
    parser.add_option("--dual", action="store_true", dest="dual", default=False)

    parser.add_option("--n_seeds", type="int", dest="n_seeds")
    
    (options, args) = parser.parse_args()
 
    return options

def main(logs_dir='YOUR_PATH_HERE'):
    
    options = parse_args()
    print(options)

    use_sgd = options.sgd
    use_dual = options.dual

    if use_sgd and not use_dual:
        print("ERROR: This script not yet set up to run sinkhorn sgd")
        return

    n_seeds = options.n_seeds
    
    try:
        os.makedirs(logs_dir)
    except:
        pass
    
    np.random.seed(1)
    
    #seeds = np.random.choice(100000, n_seeds, replace=False)
    #seeds = [18575]  # Our first hyperparameter selection run
    seeds = np.load("adult_seeds.npz")['seeds']
    #seeds = [58185]
    
    # sgd parameters
    sd_init_grid = [0.1]
    epoch_grid = [200]
    momentum_grid = [0.9]
    lr_grid = [0.001, 0.0001] #2

    # xgboost parameters
    lambda_grid = [0.000001, 0.0001, 0.01] #3
    depth_grid = [6,8,10,12,14] #5
    eta_grid = [0.005, 0.001] #2
    weight_grid = [0.1, 1.] #2 store as fraction of default weight
    pos_grid = [0.0] #1,  store as offset
    n_iter = 200


    # individual fairness parameters
    eps_grid = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] #9
    gamma_grid = [0.00005]


    # parameters for script testing
    #lr_grid = [0.01, 0.001, 0.0001]
    #lambda_grid = [0.000001]
    #depth_grid = [5]
    #eta_grid = [0.005]
    #weight_grid = [0.1] # store as fraction of default weight
    #pos_grid = [ 0.0] # store as offset
    #n_iter = 200
    #eps_grid = [0.8]
    #gamma_grid = [0.00005]

    
    hypers = [eps_grid, depth_grid, eta_grid, weight_grid, lambda_grid, pos_grid]
    names = ['eps', 'depth', 'eta', 'weight', 'lamb', 'pos']
    if not use_dual:
        hypers += [gamma_grid]
        names += ['gamma_reg']

    if use_sgd:
        hypers += [sd_init_grid, epoch_grid, momentum_grid, lr_grid]
        names += ['init', 'epoch', 'momentum', 'lr']

    names += ['seed']
    
    for seed in seeds:
        for pack in product(*hypers):
            values = list(pack)
            values.append(seed)

            if use_dual:
                (
                    eps,
                    max_depth,
                    eta,
                    min_child_weight,
                    lambda_reg,
                    scale_pos_weight
                ) = pack[:6]

            else:
                (
                    eps,
                    max_depth,
                    eta,
                    min_child_weight,
                    lambda_reg,
                    scale_pos_weight,
                    gamma_reg
                ) = pack[:7]

            if use_sgd:
                (
                    init,
                    epoch,
                    momentum,
                    lr
                ) = pack[-4:]

            exp_descriptor = []
            for n, v in zip(names, values):
                exp_descriptor.append(':'.join([n,str(v)]))
                
            exp_name = '_'.join(exp_descriptor)
            print(exp_name)

            job_cmd='python ' +\
                    'run_training.py ' +\
                    ' --eps ' + str(eps) +\
                    ' --max_depth ' + str(max_depth) +\
                    ' --eta_lr ' + str(eta) +\
                    ' --min_child_weight ' + str(min_child_weight) +\
                    ' --lambda ' + str(lambda_reg) +\
                    ' --scale_pos_weight ' + str(scale_pos_weight) +\
                    ' --n_iter ' + str(n_iter) +\
                    ' --seed ' + str(seed)

            if use_dual: job_cmd += ' --dual'
            else: job_cmd += ' --gamma ' + str(gamma_reg)

            if use_sgd: 
                job_cmd += ' --sgd --batch_size 100' +\
                        ' --sgd_init ' + str(init) +\
                        ' --epoch ' + str(epoch) +\
                        ' --momentum ' + str(momentum) +\
                        ' --lr ' + str(lr)

            

            call(['sbatch', '--nodes=1', '--account=XXX', 
                '--ntasks-per-node=4',
                '--cpus-per-task=1', '--time=00:45:00', '--mem=40GB', 
                '--partition=gpu', '--gres=gpu:1', 
                '--output=' + logs_dir + exp_name + '.out',
                '--error=' + logs_dir + exp_name + '.err',
                '--wrap', job_cmd])
    return

if __name__ == '__main__':
    main()
