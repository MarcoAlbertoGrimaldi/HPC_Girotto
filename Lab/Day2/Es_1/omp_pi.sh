#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR

module load intel/18.4

if [ -f Results/res_* ]; then rm Results/res_*; fi

for i in 1 2 4 8 16 20
    do
    for j in {0..19}
        do
	    OMP_NUM_THREADS=$i  ./omp_pi_c.x >> res_c.txt
        done
    echo "" >> res_c.txt
    done

for i in 1 2 4 8 16 20
    do
    for j in {0..19}
        do
	    OMP_NUM_THREADS=$i  ./omp_pi_a.x >> res_a.txt
        done
    echo "" >> res_a.txt
    done

for i in 1 2 4 8 16 20
    do
    for j in {0..19}
        do
	    OMP_NUM_THREADS=$i  ./omp_pi_r.x >> res_r.txt
        done
    echo""  >> res_r.txt
    done

mv res* Results/
