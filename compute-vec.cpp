// $ g++ -fopenmp -std=c++11 -O3 -march=native compute-vec.cpp && ./a.out -n 1000000000
// $ g++ -std=c++11 -O3 -march=native compute-vec.cpp -ftree-vectorize -fopt-info-vec-optimized && ./a.out -n 1000000000
// note: -fopenmp flag interferes with intrinsic vectorization

#include <stdio.h>
#include "utils.h"
#include <immintrin.h>
#include "intrin-wrapper.h"

#define VEC_LEN 4

void compute_fn0(double* A, double* B, double* C) { // Implicit vectorization
  //#pragma unroll // does not gaurantee vectorization
  //#pragma GCC ivdep // compiler specific pragma (Ignore Vector Dependency)
  //#pragma omp simd aligned(A, B, C:64) safelen(4) // requires OpenMP-4
  for (int k = 0; k < VEC_LEN; k++) {
    A[k] = A[k] * B[k] + C[k];
  }
}

void compute_fn1(double* A, double* B, double* C) { // Explicit vectorization using intrinsics
#if defined(__AVX__)
  if (VEC_LEN == 4) {
    __m256d A_ = _mm256_load_pd(A);
    __m256d B_ = _mm256_load_pd(B);
    __m256d C_ = _mm256_load_pd(C);
    A_ = _mm256_add_pd(_mm256_mul_pd(A_, B_), C_);
    _mm256_store_pd(A, A_);
  }
#else // if AVX instructions are not available, then fall back to regular C/C++ implementation
  for (int k = 0; k < VEC_LEN; k++) {
    A[k] = A[k] * B[k] + C[k];
  }
#endif
}

void compute_fn2(double* A, double* B, double* C) {
  using Vector = Vec<double, VEC_LEN>;
  Vector A_ = Vector::LoadAligned(A);
  Vector B_ = Vector::LoadAligned(B);
  Vector C_ = Vector::LoadAligned(C);
  A_ = A_ * B_ + C_;
  A_.StoreAligned(A);
}

int main(int argc, char** argv) {
  Timer t;
  long repeat = read_option<long>("-n", argc, argv);
  alignas(sizeof(double)*VEC_LEN) double A[VEC_LEN];
  alignas(sizeof(double)*VEC_LEN) double B[VEC_LEN];
  alignas(sizeof(double)*VEC_LEN) double C[VEC_LEN];
  for (int k = 0; k < VEC_LEN; k++) {
    A[k] = 1;
    B[k] = 1./(2+k);
    C[k] = 2.+k;
  }

  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn0(A, B, C);
  printf("time = %f\n", t.toc());
  printf("flop-rate = %f Gflop/s\n\n", VEC_LEN*2*repeat/1e9/t.toc());

  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn1(A, B, C);
  printf("time = %f\n", t.toc());
  printf("flop-rate = %f Gflop/s\n\n", VEC_LEN*2*repeat/1e9/t.toc());

  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn2(A, B, C);
  printf("time = %f\n", t.toc());
  printf("flop-rate = %f Gflop/s\n\n", VEC_LEN*2*repeat/1e9/t.toc());

  double sum = 0;
  for (int k = 0; k < VEC_LEN; k++) sum += A[k];
  return sum;
}

// Synposis
//
// This computation is designed to allow vectorization but only allows one AVX
// (4-wide) vector FMA instruction to execute at once. Therefore, the processor
// cannot utilize pipeling and multiple exection units (superscalar).
