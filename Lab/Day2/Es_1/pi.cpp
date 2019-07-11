#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <math.h>

int main(){
    const double a = 0;
    double const b = 1;
    unsigned long int n = 20*16*8*4*2*pow(10,5);

    double h = ( b - a ) / n;
    double h_2 = h/2;
    double x_i = 0;
    double approx = 0;

    double start = omp_get_wtime();
    double stop = 0;

    for(long int i = 1; i < n; ++i){
        x_i = a + 2 * i * h_2 + h_2;
        approx += 1/(1 + x_i*x_i);
    }

approx = 4 * approx * h;

stop = omp_get_wtime();
std::cout << "elapsed time: " << stop-start << "\n";
std::cout << "pi approximation with " << n << " iteration is:" << approx << "\n";

return 0;
}

