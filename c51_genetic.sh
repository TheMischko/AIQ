#!/bin/bash
#PBS -N C51_Genetic_Alg
#PBS -l select=1:ncpus=24:mem=32gb:scratch_local=1gb
#PBS -l walltime=95:59:00
#PBS -m ae

DATADIR=/auto/vestec1-elixir/home/dvom24

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/c51_genetic_output.txt

module add python
module add python36-modules-gcc

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input file "meta_test.py" to scratch directory
# if the copy operation fails, issue error message and exit
# cp $DATADIR/meta_test.py  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cd $DATADIR
mkdir "logs"

# move into scratch directory
cd $SCRATCHDIR

git clone https://github.com/TheMischko/AIQ.git || { echo >&2 "Error while copying git repository!"; exit 2; }

echo "Git repository was successfully cloned." >> $DATADIR/c51_genetic_output.txt

cd AIQ/agents/neural_utils/genetic
mkdir "logs"

echo "Starting python script." >> $DATADIR/c51_genetic_output.txt

python test_genetic.py -p 12 -n 4 -e 10 -i 2000 -s 1000 -a 4 -t 7 --agent_type C51 >> $DATADIR/c51_genetic_output.txt 2> $DATADIR/c51_genetic_error.txt

echo "Script finished." >> $DATADIR/c51_genetic_output.txt

cp -r "logs/" $DATADIR/ 2> $DATADIR/c51_genetic_error.txt || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch