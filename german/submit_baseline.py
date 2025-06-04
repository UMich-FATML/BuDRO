import os
import numpy as np
from subprocess import call


# NOTE: This is current set up to test a projected version!
def main(logs_dir="YOUR_LOGGING_DIRECTORY_HERE"):
    try:
        os.makedirs(logs_dir)
    except:
        pass


    seeds = np.load('german-seeds.npz')['seeds']
    #seeds = [seeds[2]]
    for seed in seeds:


        job_cmd='python ' +\
                'german_baseline.py '+\
                ' --seed ' + str(seed) +\
                ' --project'
                # remove the project flag to look at the baseline

        # an example job submission using slurm/sbatch
        call(['sbatch', '--nodes=1', '--account=XXX', '--ntasks-per-node=4',
            '--cpus-per-task=1', '--time=02:00:00', '--mem=10GB', 
            '--partition=standard',
            '--output=' + logs_dir + 'seed-' + str(seed) + '.out',
            '--error=' + logs_dir + 'seed-' + str(seed) + '.err',
            '--wrap', job_cmd])
    return

if __name__ == '__main__':
    main()
