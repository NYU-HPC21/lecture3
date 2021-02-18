// $ g++ -std=c++11 -O3 -march=native nbody.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "intrin-wrapper.h"

#define CLOCK_FREQ 3.3e9

double rsqrt0(double r2) {
  return 1./sqrt(r2);
}

double rsqrt1(double r2) { // Fast inverse square root used in Quake III
  union {
    int64_t i;
    double rinv;
  };
  rinv = r2;
  i = 0x5FE6EB50C7B537A9 - (i >> 1);

  // Newton iteration for higher accuracy
  rinv *= ((3.0) - r2 * rinv * rinv) * 0.5;
  return rinv;
}

template <double(*RSQRT_FN)(double)> void kernel(long N, const double* x, const double* y, const double* z, const double* f, double* u) {
  for (long trg = 0; trg < N; trg++) { // loop over targets
    double tx = x[trg];
    double ty = y[trg];
    double tz = z[trg];
    double potential = 0;
    for (long src = 0; src < N; src++) { // loop over sources
      double dx = tx - x[src];
      double dy = ty - y[src];
      double dz = tz - z[src];
      double r2 = dx*dx + dy*dy + dz*dz;

      double rinv = (r2 > 0 ? RSQRT_FN(r2) : 0);
      potential += rinv * f[src]; // sum potential
    }
    u[trg] = potential;
  }
}

void kernel_vec(long N, const double* x, const double* y, const double* z, const double* f, double* u){
#if defined(__AVX__)
  constexpr int VecLen = 4;
  __m256d zero = _mm256_set1_pd(0);
  __m256d three = _mm256_set1_pd(3);
  __m256d half = _mm256_set1_pd(0.5);
  for (long trg = 0; trg < N; trg += VecLen) { // loop over targets
    __m256d tx = _mm256_load_pd(x + trg);
    __m256d ty = _mm256_load_pd(y + trg);
    __m256d tz = _mm256_load_pd(z + trg);
    __m256d potential = _mm256_setzero_pd();
    for (long src = 0; src < N; src++) { // loop over sources
      __m256d dx = _mm256_sub_pd(tx, _mm256_broadcast_sd(x + src));
      __m256d dy = _mm256_sub_pd(ty, _mm256_broadcast_sd(y + src));
      __m256d dz = _mm256_sub_pd(tz, _mm256_broadcast_sd(z + src));
      __m256d r2 = _mm256_add_pd(_mm256_mul_pd(dx,dx), _mm256_add_pd(_mm256_mul_pd(dy,dy), _mm256_mul_pd(dz,dz)));

      __m256d rinv = _mm256_and_pd(_mm256_cmp_pd(r2, zero, _CMP_GT_OS), _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2))));
      rinv = _mm256_mul_pd(rinv, _mm256_mul_pd(_mm256_sub_pd(three, _mm256_mul_pd(r2, _mm256_mul_pd(rinv, rinv))), half)); // Newton iteration
      potential = _mm256_add_pd(potential, _mm256_mul_pd(rinv, _mm256_broadcast_sd(f + src)));
    }
    _mm256_store_pd(u + trg, potential);
  }
#elif defined(__SSE4_2__)
  constexpr int VecLen = 2;
  __m128d zero = _mm_set1_pd(0);
  __m128d three = _mm_set1_pd(3);
  __m128d half = _mm_set1_pd(0.5);
  for (long trg = 0; trg < N; trg += VecLen) { // loop over targets
    __m128d tx = _mm_load_pd(x + trg);
    __m128d ty = _mm_load_pd(y + trg);
    __m128d tz = _mm_load_pd(z + trg);
    __m128d potential = _mm_setzero_pd();
    for (long src = 0; src < N; src++) { // loop over sources
      __m128d dx = _mm_sub_pd(tx, _mm_load1_pd(x + src));
      __m128d dy = _mm_sub_pd(ty, _mm_load1_pd(y + src));
      __m128d dz = _mm_sub_pd(tz, _mm_load1_pd(z + src));
      __m128d r2 = _mm_add_pd(_mm_mul_pd(dx,dx), _mm_add_pd(_mm_mul_pd(dy,dy), _mm_mul_pd(dz,dz)));

      __m128d rinv = _mm_and_pd(_mm_cmpgt_pd(r2, zero), _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(r2))));
      rinv = _mm_mul_pd(rinv, _mm_mul_pd(_mm_sub_pd(three, _mm_mul_pd(r2, _mm_mul_pd(rinv, rinv))), half)); // Newton iteration
      potential = _mm_add_pd(potential, _mm_mul_pd(rinv, _mm_load1_pd(f + src)));
    }
    _mm_store_pd(u + trg, potential);
  }
#else
  kernel<rsqrt0>(N, x, y, z, f, u);
#endif
}

void kernel_vec_(long N, const double* x, const double* y, const double* z, const double* f, double* u){
  constexpr int VecLen = 4;
  constexpr double zero = 0;
  using Vector = Vec<double, VecLen>;
  for (long trg = 0; trg < N; trg += VecLen) { // loop over targets
    Vector tx = Vector::LoadAligned(x + trg);
    Vector ty = Vector::LoadAligned(y + trg);
    Vector tz = Vector::LoadAligned(z + trg);
    Vector potential = Vector::Zero();
    for (long src = 0; src < N; src++) { // loop over sources
      Vector dx = tx - Vector::Load1(x + src);
      Vector dy = ty - Vector::Load1(y + src);
      Vector dz = tz - Vector::Load1(z + src);
      Vector r2 = dx*dx + dy*dy + dz*dz;

      Vector rinv = (r2 > zero) & approx_rsqrt(r2);
      rinv *= ((3.0) - r2 * rinv * rinv) * 0.5; // Newton iteration
      potential += rinv * Vector::Load1(f + src);
    }
    potential.StoreAligned(u + trg);
  }
}

int main(int argc, char** argv) {
  long N = 10000;
  Timer t;

  double* x = (double*) aligned_malloc(N * sizeof(double));
  double* y = (double*) aligned_malloc(N * sizeof(double));
  double* z = (double*) aligned_malloc(N * sizeof(double));
  double* f = (double*) aligned_malloc(N * sizeof(double));
  double* u0 = (double*) aligned_malloc(N * sizeof(double));
  double* u1 = (double*) aligned_malloc(N * sizeof(double));
  double* u2 = (double*) aligned_malloc(N * sizeof(double));

  for (long i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    f[i] = drand48();
    u0[i] = 0;
    u1[i] = 0;
    u2[i] = 0;
  }

  t.tic();
  kernel<rsqrt0>(N, x, y, z, f, u0);
  printf("Reference implementation: %f cycles/eval\n", t.toc()*CLOCK_FREQ/(N*N));

  t.tic();
  kernel<rsqrt1>(N, x, y, z, f, u1);
  printf("Quake III implementation: %f cycles/eval\n", t.toc()*CLOCK_FREQ/(N*N));

  t.tic();
  kernel_vec_(N, x, y, z, f, u2);
  printf("Vectorized implementation: %f cycles/eval\n\n", t.toc()*CLOCK_FREQ/(N*N));

  double max_val = 0, max_err1 = 0, max_err2 = 0;
  for (long i = 0; i < N; i++) {
    max_val = std::max(max_val, fabs(u0[i]));
    max_err1 = std::max(max_err1, fabs(u1[i] - u0[i]));
    max_err2 = std::max(max_err2, fabs(u2[i] - u0[i]));
  }
  printf("Relative error (Quake III) = %e\n", max_err1 / max_val);
  printf("Relative error (Vectorized) = %e\n", max_err2 / max_val);

  return 0;
}

// Synopsis
//
// Special functions like trigonometric functions, sqrt, 1/r etc. are either
// imeplemented in software or have dedicated special hardware units (which can
// still be very slow) and are harder to vectorize.
//
// This example shows how fast reciprocal-sqrt function (1/sqrt(r)) can be
// optimized. This kernel is used in astro-physics calculations to compute
// gravitational fields, in fluid-dynamics and compute graphics.
//
// * The rsqrt1 function uses the approximate implementation developed for Quake
// III video game. (https://en.wikipedia.org/wiki/Fast_inverse_square_root)
// This method gives about 2-digits of accuracy. Newton iteration can be used to
// get higher accuracy.
//
// * The function rsqrt_vec uses an approximate reciprocal-sqrt instruction
// along with Newton iterations and is vectorized using intrinsics.

