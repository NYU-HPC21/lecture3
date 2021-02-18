// $ g++ -fopenmp -std=c++11 -O3 -march=native compute-vec-pipe.cpp && ./a.out -n 1000000000
// $ g++ -std=c++11 -O3 -march=native compute-vec-pipe.cpp -ftree-vectorize -fopt-info-vec-optimized && ./a.out -n 1000000000
// note: -fopenmp flag interferes with intrinsic vectorization

#include <stdio.h>
#include "utils.h"
#include <immintrin.h>
#include "intrin-wrapper.h"

#define VEC_LEN 4
constexpr int M = 1;

void compute_fn0(double* A, double* B, double* C) { // Implicit vectorization
  //#pragma unroll // does not gaurantee vectorization
  //#pragma GCC ivdep // compiler specific pragma (Ignore Vector Dependency)
  //#pragma omp simd aligned(A, B, C:64) safelen(4) // requires OpenMP-4
  for (int k = 0; k < M*VEC_LEN; k++) {
    A[k] = A[k] * B[k] + C[k];
  }
}

void compute_fn1(double* A, double* B, double* C) { // Explicit vectorization using intrinsics
#if defined(__AVX__)
  if (VEC_LEN == 4) {
    for (int k = 0; k < M; k++) {
      __m256d A_ = _mm256_load_pd(A + k * VEC_LEN);
      __m256d B_ = _mm256_load_pd(B + k * VEC_LEN);
      __m256d C_ = _mm256_load_pd(C + k * VEC_LEN);
      A_ = _mm256_add_pd(_mm256_mul_pd(A_, B_), C_);
      _mm256_store_pd(A + k * VEC_LEN, A_);
    }
  }
#elif defined(__SSE2__)
  if (VEC_LEN == 4) {
    for (int k = 0; k < 2*M; k++) {
      __m128d A_ = _mm_load_pd(A + k * 2);
      __m128d B_ = _mm_load_pd(B + k * 2);
      __m128d C_ = _mm_load_pd(C + k * 2);
      A_ = _mm_add_pd(_mm_mul_pd(A_, B_), C_);
      _mm_store_pd(A + k * 2, A_);
    }
  }
#else // if AVX instructions are not available, then fall back to regular C/C++ implementation
  for (int k = 0; k < VEC_LEN*M; k++) {
    A[k] = A[k] * B[k] + C[k];
  }
#endif
}

void compute_fn2(double* A, double* B, double* C) {
  using Vector = Vec<double, VEC_LEN>;
  for (int k = 0; k < M; k++) {
    Vector A_ = Vector::LoadAligned(A + k * VEC_LEN);
    Vector B_ = Vector::LoadAligned(B + k * VEC_LEN);
    Vector C_ = Vector::LoadAligned(C + k * VEC_LEN);
    A_ = A_ * B_ + C_;
    A_.StoreAligned(A + k * VEC_LEN);
  }
}

int main(int argc, char** argv) {
  Timer t;
  long repeat = read_option<long>("-n", argc, argv);
  alignas(sizeof(double)*VEC_LEN) double A[VEC_LEN*M];
  alignas(sizeof(double)*VEC_LEN) double B[VEC_LEN*M];
  alignas(sizeof(double)*VEC_LEN) double C[VEC_LEN*M];
  for (int k = 0; k < VEC_LEN*M; k++) {
    A[k] = 1;
    B[k] = 1./(2+k);
    C[k] = 2.+k;
  }

  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn0(A, B, C);
  printf("time = %f\n", t.toc());
  printf("flop-rate = %f Gflop/s\n\n", M*VEC_LEN*2*repeat/1e9/t.toc());

  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn1(A, B, C);
  printf("time = %f\n", t.toc());
  printf("flop-rate = %f Gflop/s\n\n", M*VEC_LEN*2*repeat/1e9/t.toc());

  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn2(A, B, C);
  printf("time = %f\n", t.toc());
  printf("flop-rate = %f Gflop/s\n\n", M*VEC_LEN*2*repeat/1e9/t.toc());

  double sum = 0;
  for (int k = 0; k < VEC_LEN*M; k++) sum += A[k];
  return sum;
}

// Synposis
//
// In this example the parameter M controls the number of independent FMA
// instructions that execute at the same time and allows the processor to
// pipeline instructions and use multiple execution units (superscalar). This
// should give close to the peak theoretial performance.
//
// * Measure the Flop-rate for different values of M and compare it with the
// peak theoretial performance.
// * If M is too small then there isn't enough parallelism to fill the pipeline.
// * If M is too large then the computation cannot fit in registers and the data
// spills to the L1 cache resulting in slower execution.
