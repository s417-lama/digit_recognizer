#ifndef OPTIMIZE_FUNC_H
#define OPTIMIZE_FUNC_H

int f_dimension();
double f_value(const double w[]);
void f_gradient(const double w[], double g[]);
double evaluate(const double w[]);
void init();

#endif
