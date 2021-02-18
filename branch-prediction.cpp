// $ g++ -O3 branch-prediction.cpp && ./a.out

#include <stdio.h>
#include <algorithm>
#include "utils.h"

int main(int argc, char** argv) {
  long N = 32768;
  int* data = (int*) malloc(N * sizeof(int));
  for (long i = 0; i < N; i++) data[i] = std::rand() % 256;

  // Try with and without sorting
  std::sort(data, data + N);

  Timer t;
  t.tic();
  long long sum = 0;
  for (long i = 0; i < 10000; i++) {
    for (long j = 0; j < N; j++) {
      if (data[j] >= 128) sum += data[i];
    }
  }
  printf("time = %f\n", t.toc());

  return sum;
}

