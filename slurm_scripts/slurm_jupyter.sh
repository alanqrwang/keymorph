#!/bin/bash
#
#SBATCH --job-name=jupyter # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=50:30:00 # set this time according to your need
#SBATCH --mem=32GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:a100:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
module purge
module load miniconda3/22.11.1-ctkwnpe
# if using conda
source activate keymorph
# if using pip
# source ~/myvev/bin/activate

# set log file
LOG="/home/${USER}/keymorph/job_out/jupyterjob_${SLURM_JOB_ID}.txt"

# gerenerate random port
PORT=`shuf -i 10000-50000 -n 1`

# print useful info
cat << EOF > ${LOG}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~ Slurm Job $SLURM_JOB_ID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hello from the Jupyter job!
In order to connect to this jupyter session

1. setup a tunnel on your local workstation with:
     --->  ssh -t ${USER}@ai-login01.med.cornell.edu -L ${PORT}:localhost:${PORT} ssh ${HOSTNAME} -L ${PORT}:localhost:${PORT}
(copy above command and paste it to your terminal).
Depending on your ssh configuration, you may be
prompted for your password. Once you are logged in,
leave the terminal running and don't close it until
you are finished with your Jupyter session. 

2. Further down look for a line similar to
     ---> http://127.0.0.1:10439/?token=xxxxyyyxxxyyy
Copy this line and paste in your browser 
EOF

# start jupyter
jupyter-notebook --no-browser --ip=0.0.0.0 --port=${PORT} 2>&1 | tee -a ${LOG}

