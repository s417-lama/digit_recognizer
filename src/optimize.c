#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "optimize.h"

double calc_norm(const int dim, double v[])
{
  double tmp = 0;
  int i = 0;
  for (i = 0; i < dim; i++) {
    tmp += v[i] * v[i];
  }
  tmp = sqrt(tmp);
  return tmp;
}

int optimize(const double alpha, const int dim, double x[], 
             void (*calc_grad)(const double [], double []),
             double (*calc_value)(const double[]), int N)
{
  int i;

  double *g = malloc(dim * sizeof(double));

  int iter = 0;
  while (++iter <= N) {

    (*calc_grad)(x, g);

    if (iter % 1000 == 0) printf("%6d / %d\n", iter, N);

    for (i = 0; i < dim; i++) {
      x[i] -= alpha * g[i];
    }
  }

  free(g);

  return iter;
}
