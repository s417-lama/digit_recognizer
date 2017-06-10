#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "optimize.h"
#include "func.h"

int main(const int argc, const char **argv)
{
  srand((unsigned int)time(NULL));
  init();
  
  const double alpha = (argc >= 2) ? atof(argv[1]) : 0.01;
  const int N = (argc == 3) ? atoi(argv[2]) : 60000;

  int i;
  const int dim = f_dimension();

  double *x = malloc(dim * sizeof(double));
  for (i = 0; i < dim; i++) {
    x[i] = (double)rand() / RAND_MAX - 0.5;
  }

  printf("alpha = %f\n", alpha);

  optimize(alpha, dim, x, f_gradient, f_value, N);
  
  double p = evaluate(x);
  printf("identification rate: %.4f\n", p);
  
  free(x);

  return 0;
}

