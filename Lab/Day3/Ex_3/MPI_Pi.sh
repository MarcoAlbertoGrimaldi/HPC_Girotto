#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR

module load openmpi/1.8.3/intel/14.0

if [ -f res_* ]; then rm res_*; fi

for i in 1 2 4 8 16 24 32 40
    do
    for j in {1..19}
        do
	    mpirun -np $i ./ex_3.x >> res.txt
        done
    echo "" >> res.txt
    done
