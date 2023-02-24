#!/bin/bash
#PBS -N DeepQL_Genetic_Alg
#PBS -l select=1:ncpus=24:mem=32gb:scratch_local=1gb
#PBS -l walltime=72:00:00
#PBS -m ae

DATADIR=/auto/vestec1-elixir/home/dvom24

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add python
module add python36-modules-gcc

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input file "meta_test.py" to scratch directory
# if the copy operation fails, issue error message and exit
# cp $DATADIR/meta_test.py  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cd $DATADIR

# move into scratch directory
cd $SCRATCHDIR

git clone https://github.com/TheMischko/AIQ.git || { echo >&2 "Error while copying git repository!"; exit 2; }

echo "Git repository was successfully cloned." >> $DATADIR/jobs_info.txt

cd AIQ/

echo "Starting python script." >> $DATADIR/test_set_params_out.txt

python test_set_params >> $DATADIR/test_set_params_out.txt 2> $DATADIR/test_set_params_err.txt

# clean the SCRATCH directory
clean_scratch