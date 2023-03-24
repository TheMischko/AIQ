#!/bin/bash
#PBS -N DeepQL_100k
#PBS -l select=1:ncpus=8:mem=12gb:scratch_local=1gb
#PBS -l walltime=167:59:00
#PBS -m ae

DATADIR=/auto/vestec1-elixir/home/dvom24

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/DeepQL_test_output.txt

module add python
module add python36-modules-gcc

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input file "meta_test.py" to scratch directory
# if the copy operation fails, issue error message and exit
# cp $DATADIR/meta_test.py  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cd $DATADIR
mkdir "log"

# move into scratch directory
cd $SCRATCHDIR

git clone https://github.com/TheMischko/AIQ.git || { echo >&2 "Error while copying git repository!"; exit 2; }

echo "Git repository was successfully cloned." >> $DATADIR/DeepQL_test_output.txt

mkdir "log"

echo "Starting python script." >> $DATADIR/DeepQL_test_output.txt

python AIQ.py --log --verbose_log_el --save_samples -r BF -l 100000 -s 10000 -t 8 -a DeepQL,0.00468,0.33,32,3000,64,224,176,0.25,60 >> $DATADIR/DeepQL_test_output.txt 2> $DATADIR/DeepQL_test_error.txt

echo "Script finished." >> $DATADIR/DeepQL_test_output.txt

cp -r "log/" $DATADIR/ 2> $DATADIR/DeepQL_test_error.txt || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch