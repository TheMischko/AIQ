#!/bin/bash
#PBS -N DeepQL_100k
#PBS -l select=1:ncpus=8:mem=12gb:scratch_local=1gb
#PBS -l walltime=23:59:00

DATADIR=/auto/vestec1-elixir/home/dvom24

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/DeepQL_test_output.txt

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
echo "Git repository was successfully cloned." >> $DATADIR/DeepQL_test_output.txt

# prepare scratch directory
mkdir -p "log"
mkdir -p "log-el"

# run script
echo "Starting python script." >> $DATADIR/DeepQL_test_output.txt
python AIQ.py --log --verbose_log_el -r BF -l 100000 -s 10000 -t 8 -a DeepQL,0.00468,0.33,32,3000,64,224,176,0.25,60 >> $DATADIR/DeepQL_test_output.txt 2> $DATADIR/DeepQL_test_error.txt
echo "Script finished." >> $DATADIR/DeepQL_test_output.txt

# copy output files to DATADIR
cp -r "log/*" $DATADIR/log 2> $DATADIR/DeepQL_test_error.txt || { echo >&2 "Log file(s) copying failed (with a code $?) !!"; exit 4; }
cp -r "log-el/*" $DATADIR/log-el 2> $DATADIR/DeepQL_test_error.txt || { echo >&2 "Log-el file(s) copying failed (with a code $?) !!"; exit 5; }

# clean the SCRATCH directory
clean_scratch