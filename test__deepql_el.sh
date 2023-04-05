#!/bin/bash
#PBS -N DeepQL_EL_3k
#PBS -l select=1:ncpus=8:mem=12gb:scratch_local=1gb
#PBS -l walltime=167:59:00
#PBS -m ae

DATADIR=/auto/vestec1-elixir/home/dvom24

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/DeepQL_EL_test_output.txt

# add modules
module add python36-modules-gcc
# test SCRATCHDIR
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# prepare data directory
cd $DATADIR
mkdir -p "log"
mkdir -p "log-el"

# move into scratch directory
cd $SCRATCHDIR

# Clone repository and move into it
git clone https://github.com/TheMischko/AIQ.git || { echo >&2 "Error while copying git repository!"; exit 2; }
cd AIQ
echo "Git repository was successfully cloned." >> $DATADIR/DeepQL_EL_test_output.txt

# prepare scratch directory
mkdir -p "log"
mkdir -p "log-el"

# run script
echo "Starting python script." >> $DATADIR/DeepQL_EL_test_output.txt
# DQL_Dual_ET_Decay,0.04,0.34,16,290,200,144,0,0.05,10,0.9,0
python AIQ.py --log --verbose_log_el -r BF -l 100000 -s 1000 -t 8 -a DQL_Dual_ET_Decay,0.00468,0.33,32,3000,64,224,176,0.25,60,0.7,0 >> $DATADIR/DeepQL_EL_test_output.txt 2> $DATADIR/DeepQL_EL_test_error.txt
echo "Script finished." >> $DATADIR/DeepQL_EL_test_output.txt

# copy output files to DATADIR
echo "Content of SCRATCH:"
ls
echo "------------------------"
echo "Content of log:"
cd log
ls
cd ..
echo "------------------------"
echo "Content of log-el:"
cd log-el
ls
cd ..
echo "------------------------"
echo "Copying log/"
cp -r "log" $DATADIR 2> $DATADIR/DeepQL_EL_test_error.txt || { echo >&2 "Log file(s) copying failed (with a code $?) !!"; exit 4; }
echo "Copying log-el/"
cp -r "log-el" $DATADIR 2> $DATADIR/DeepQL_EL_test_error.txt || { echo >&2 "Log-el file(s) copying failed (with a code $?) !!"; exit 5; }

# clean the SCRATCH directory
clean_scratch