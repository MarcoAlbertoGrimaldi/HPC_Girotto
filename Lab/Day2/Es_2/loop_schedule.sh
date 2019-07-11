#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR

module load intel/18.4

if [ -f res_* ]; then rm res_*; fi

MP_NUM_THREADS=10  ./loop_schedule.x >> res_loop.txt

