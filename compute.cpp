// $ g++ -std=c++11 -O3 -march=native compute.cpp && ./a.out
// $ g++ -std=c++11 -O3 -march=native compute.cpp -ftree-vectorize -fopt-info-vec-optimized && ./a.out -n 1000000000

#include <stdio.h>
#include <math.h>
#include "utils.h"

#define CLOCK_FREQ 3.3e9

void compute_fn(double* A, double B, double C) {
  (*A) = (*A) * B + C;
  //(*A) = C / (*A);
  //(*A) = sqrt(*A);
  //(*A) = sin(*A);
}

int main(int argc, char** argv) {
  Timer t;
  long repeat = read_option<long>("-n", argc, argv);
  double A = 1.5;
  double B = 1./2;
  double C = 2.;

  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn(&A, B, C);
  printf("%f seconds\n", t.toc());
  printf("%f cycles/eval\n", t.toc()*CLOCK_FREQ/repeat);
  printf("%f Gflop/s\n", 2*repeat/1e9/t.toc());

  return A;
}

// Synopsis
//
// By design, this computation is such that only one fused-multiply-accumulate
// instruction can execute at one time i.e. this computation cannot be
// vectorized or pipelined. Therefor, this example can be used to measure
// latency of operations.
//
// * Compare the observed latency with the expected latency for _mm256_fmadd_pd
// instruction for your architecture from this link:
// (https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmadd_pd&expand=2508)
//
// * Try replacing the mult-add operation with some other computation, like
// division, sqrt, sin, cos etc. to measure the latency of those operations and
// compare it with the latency of mult-add.
