import os
import numpy as np
from subprocess import call
from optparse import OptionParser
from itertools import product

# How you would submit many hyperparameters for fair training on German

def parse_args():
    
    parser = OptionParser()
    parser.set_defaults()
    
    parser.add_option("--dual", action="store_true", dest="dual", default=False)
    parser.add_option(
            "--include_age_bin", action="store_true", dest="agebin", default=False
            )
    (options, args) = parser.parse_args()
 
    return options

def main(logs_dir="YOUR_LOGGING_DIRECTORY_HERE"):
    try:
        os.makedirs(logs_dir)
    except:
        pass

    options = parse_args()
    print(options)

    use_dual = options.dual
    agebin = options.agebin

    #eps_grid = [0.6, 0.8, 1.0, 1.2, 1.4]
    eps_grid = [0.5, 1.0, 1.5, 2.0, 2.5]
    depth_grid = [4,7,10,13] #4
    eta_grid = [0.0001, 0.001, 0.005, 0.01, 0.05 ] #5

    # test everything
    #eps_grid = [2.0]
    #depth_grid = [4]
    #eta_grid = [0.05]

    hypers = [eps_grid, depth_grid, eta_grid]

    #seeds = np.load('german-seeds.npz')['seeds']
    seeds = [seeds[-1]]
    for seed in seeds:
        for pack in product(*hypers):

            eps = pack[0]
            depth = pack[1]
            eta = pack[2]

            name_string = "eps:{}_depth:{}_eta:{}".format(eps, depth, eta)

            job_cmd='python ' +\
                    'german_fair_age.py ' +\
                    ' --seed ' + str(seed) + ' --eps ' + str(eps) +\
                    ' --max_depth ' + str(depth) + ' --eta_lr ' + str(eta)

            if use_dual:
                job_cmd += ' --dual'

            if agebin:
                job_cmd += ' --include_age_bin'

            call(['sbatch', '--nodes=1', '--account=XXX', 
                '--ntasks-per-node=2',
                '--cpus-per-task=1', '--time=00:30:00', '--mem=10GB', 
                '--partition=standard', #'--gres=gpu:1', 
                '--output=' + logs_dir + name_string +\
                '_seed-' + str(seed)  + '.out',
                '--error=' + logs_dir + name_string +\
                '_seed-' + str(seed)  + '.err',
                '--wrap', job_cmd])
    return

if __name__ == '__main__':
    main()
