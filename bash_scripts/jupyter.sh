#!/bin/bash
#
#SBATCH --job-name=jupyter # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=50:30:00 # set this time according to your need
#SBATCH --mem=16GB # how much RAM will your notebook consume? 
#SBATCH -p scu-gpu # specify partition
#SBATCH -o ./job_out/%j-jupyter.out
#SBATCH -e ./job_err/%j-jupyter.err \ 
module purge
source /midtier/sablab/scratch/alw4013/miniconda3/bin/activate keymorph
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

