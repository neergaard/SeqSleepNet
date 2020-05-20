import os
import tempfile


def submit_job(jobname, script):
    
    content = '''#!/bin/bash
#
#SBATCH --job-name={0}
#SBATCH -p mignot,normal,owners
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/users/alexno/SeqSleepNet/logs/{0}.out
#SBATCH --error=/home/users/alexno/SeqSleepNet/logs/{0}.err
############################################################

ml load matlab
cd $HOME/SeqSleepNet/data_processing

{1}
'''
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.format(jobname, script).encode())
    os.system('sbatch {}'.format(j.name))


if __name__ == '__main__':
    
    n = 200
    base_s = 'matlab -nodisplay -r "prepare_seqsleepnet_data({0}, {1}); exit"'
    base_n = 'data-'

    for i in range(n):
        submit_job(base_n + str(i+1).zfill(3), base_s.format(i+1, n))

    print('All jobs have been submitted!')

