#!/bin/bash

if [ -f res.txt ]; then rm res.txt; fi

for i in 1 2 4 8 16 24 32 40
    do
	mpirun -np $i ./ex_3.x >> res.txt
    done
    
