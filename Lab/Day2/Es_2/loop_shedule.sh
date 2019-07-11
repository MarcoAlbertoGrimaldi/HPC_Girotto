#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR

if [ -f Results/res_* ]; then rm Results/res_*; fi

MP_NUM_THREADS=10  ./loop_schedule.x >> res_loop.txt
   
mv res* Results/