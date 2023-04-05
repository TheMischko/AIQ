#!/bin/bash
#PBS -N EL_Genetic_2_Alg
#PBS -l select=1:ncpus=24:mem=18gb:scratch_local=1gb
#PBS -l walltime=47:59:00
#PBS -m ae

DATADIR=/auto/vestec1-elixir/home/dvom24

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/DQN_EL_genetic_2_output.txt

module add python36-modules-gcc

# test if scratch directory is set
#
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cd $DATADIR
mkdir -p "logs"
mkdir -p "gen2"

# move into scratch directory
cd $SCRATCHDIR

git clone https://github.com/TheMischko/AIQ.git || { echo >&2 "Error while copying git repository!"; exit 2; }

echo "Git repository was successfully cloned." >> $DATADIR/DQN_EL_genetic_2_output.txt

cd AIQ/agents/deep_ql/neural_utils/genetic
mkdir -p "logs"

echo "Starting python script." >> $DATADIR/DQN_EL_genetic_2_output.txt

python test_genetic.py -p 12 -n 4 -e 10 -i 10000 -s 1000 -a 4 -t 8 --agent_type DQL_EL --debug >> $DATADIR/DQN_EL_genetic_2_output.txt 2> $DATADIR/DQN_EL_genetic_2_error.txt

echo "Script finished." >> $DATADIR/DQN_EL_genetic_2_output.txt

cp -r "logs" $DATADIR/gen2 2> $DATADIR/DQN_EL_genetic_2_error.txt || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch