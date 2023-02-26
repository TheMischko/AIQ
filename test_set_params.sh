#!/bin/bash
#PBS -N DeepQL_Test_Params
#PBS -l select=1:ncpus=8:mem=16gb:scratch_local=1gb
#PBS -l walltime=16:00:00

DATADIR=/auto/vestec1-elixir/home/dvom24

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/test_set_params_out.txt

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

echo "Git repository was successfully cloned." >> $DATADIR/test_set_params_out.txt

cd AIQ/ || { echo >&2 "Error while moving into AIQ dir.!"; exit 3; }

echo "Starting python script." >> $DATADIR/test_set_params_out.txt

python test_set_params.py >> $DATADIR/test_set_params_out.txt 2> $DATADIR/test_set_params_err.txt

echo "Python script finished." >> $DATADIR/test_set_params_out.txt

# clean the SCRATCH directory
clean_scratch