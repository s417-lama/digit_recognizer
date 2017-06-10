#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NTRAIN 60000
#define NTEST 10000
#define NPIXEL 784
#define NDIGIT 10
#define NHIDDEN 200

const char *train_label_data_path = "../data/mnist/train-labels-idx1-ubyte";
const char *test_label_data_path = "../data/mnist/t10k-labels-idx1-ubyte";
const char *train_image_data_path = "../data/mnist/train-images-idx3-ubyte";
const char *test_image_data_path = "../data/mnist/t10k-images-idx3-ubyte";

int *train_labels;
double **train_images;
int *test_labels;
double **test_images;

int n_samples;

int *t;
double **x;
double a1[NHIDDEN];
double a2[NDIGIT];
double z[NHIDDEN + 1]; // +1 is for bias
double y[NDIGIT];
double delta1[NHIDDEN + 1]; // +1 is for bias
double delta2[NDIGIT];

double softmax(const double d[], int i, int n)
{
  int j;
  double sum = 0;
  for (j = 0; j < n; j++) {
    sum += exp(d[j]);
  }
  return exp(d[i]) / sum;
}

double sigmoid(double d)
{
  return 1.0 / (1.0 + exp(-d));
}

double sigmoid_prime(double d)
{
  double f = sigmoid(d);
  return f * (1 - f);
}

double dot(const double c[], const double d[], int n)
{
  double r = 0;
  int i;
  for (i = 0; i < n; i++) {
    r += c[i] * d[i];
  }
  return r;
}

int maxi(const double d[], int n)
{
  int i;
  double max = 0;
  int max_i = 0;
  for (i = 0; i < n; i++) {
    if (d[i] > max) {
      max = d[i];
      max_i = i;
    }
  }
  return max_i;
}

void test_mode(int n)
{
  if (n == 1) {
    t = test_labels;
    x = test_images;
    n_samples = NTEST;
  } else if (n == 0) {
    t = train_labels;
    x = train_images;
    n_samples = NTRAIN;
  }
}

void feedforward(const double w[], double (*h)(double), int n)
{
  int i;
  
  // a1
  for (i = 0; i < NHIDDEN; i++) {
    a1[i] = dot(w, x[n], NPIXEL);
    w += NPIXEL;
    a1[i] += *w; // bias
    w++;
  }

  // z
  for (i = 0; i < NHIDDEN; i++) {
    z[i] = h(a1[i]);
  }
  z[NHIDDEN] = 1;

  // a2
  for (i = 0; i < NDIGIT; i++) {
    a2[i] = dot(w, z, NHIDDEN + 1);
    w += NHIDDEN + 1;
  }

  // y
  for (i = 0; i < NDIGIT; i++) {
    y[i] = softmax(a2, i, NDIGIT);
  }
}

void backprop(const double w[], double (*h_prime)(double), int n)
{
  int i, j;

  // delta2
  for (i = 0; i < NDIGIT; i++) {
    delta2[i] = y[i] - ((i == t[n]) ? 1 : 0);
  }

  // delta1
  w += (NPIXEL + 1) * NHIDDEN;
  for (i = 0; i < NHIDDEN + 1; i++) {
    double tmp = 0;
    const double *wp = w;
    for (j = 0; j < NDIGIT; j++) {
      tmp += *wp * delta2[j];
      wp += NHIDDEN + 1;
    }
    delta1[i] = h_prime(z[i]) * tmp;
    w++;
  }
}

double evaluate(const double w[])
{
  test_mode(1);
  int count = 0;
  int i;
  for (i = 0; i < n_samples; i++) {
    feedforward(w, sigmoid, i);
    if (maxi(y, NDIGIT) == t[i]) count++;
  }
  return (double)count / n_samples;
}

int f_dimension()
{
  // layer 1 + layer 2
  return (NPIXEL + 1) * NHIDDEN + (NHIDDEN + 1) * NDIGIT;
}

double f_value(const double w[])
{
  int i;
  double E = 0;
  for (i = 0; i < n_samples; i++) {
    feedforward(w, sigmoid, i);
    E -= log(y[t[i]]);
  }
  return E;
}

void f_gradient(const double w[], double g[])
{
  int n = rand() % n_samples;
  feedforward(w, sigmoid, n);
  backprop(w, sigmoid_prime, n);
  
  int i, j;
  
  // layer 1
  for (i = 0; i < NHIDDEN; i++) {
    for (j = 0; j < NPIXEL; j++) {
      *g = delta1[i] * x[n][j];
      g++;
    }
    *g = delta1[i];
    g++;
  }

  // layer 2
  for (i = 0; i < NDIGIT; i++) {
    for (j = 0; j < NHIDDEN + 1; j++) {
      *g = delta2[i] * z[j];
      g++;
    }
  }
}

void init()
{
  FILE *fp;
  int i, j;
  
  if ((fp = fopen(train_label_data_path, "r")) != NULL) {
    train_labels = (int *)malloc(sizeof(int) * NTRAIN);
    char buf[10];
    fread(&buf, sizeof(char), 8, fp); // skip
    char *n = (char *)malloc(sizeof(char) * NTRAIN);
    fread(n, sizeof(char), NTRAIN, fp);
    for (i = 0; i < NTRAIN; i++) {
      train_labels[i] = (int)n[i];
    }
    free(n);
    fclose(fp);
  } else {
    printf("file (%s) open failed.\n", train_label_data_path);
  }
  
  if ((fp = fopen(train_image_data_path, "r")) != NULL) {
    char buf[20];
    fread(&buf, sizeof(char), 16, fp); // skip
    train_images = (double **)malloc(sizeof(double *) * NTRAIN);
    unsigned char *pixels = (unsigned char *)malloc(sizeof(unsigned char) * NTRAIN * NPIXEL);
    fread(pixels, sizeof(unsigned char), NTRAIN * NPIXEL, fp);
    for (i = 0; i < NTRAIN; i++) {
      double *p = (double *)malloc(sizeof(double) * NPIXEL);
      for (j = 0; j < NPIXEL; j++) {
	p[j] = (double)pixels[i * NPIXEL + j] / 255.0;
      }
      train_images[i] = p;
    }
    free(pixels);
    fclose(fp);
  } else {
    printf("file (%s) open failed.\n", train_image_data_path);
  }
  
  if ((fp = fopen(test_label_data_path, "r")) != NULL) {
    test_labels = (int *)malloc(sizeof(int) * NTEST);
    char buf[10];
    fread(&buf, sizeof(char), 8, fp); // skip
    char *n = (char *)malloc(sizeof(char) * NTEST);
    fread(n, sizeof(char), NTEST, fp);
    for (i = 0; i < NTEST; i++) {
      test_labels[i] = (int)n[i];
    }
    free(n);
    fclose(fp);
  } else {
    printf("file (%s) open failed.\n", test_label_data_path);
  }
  
  if ((fp = fopen(test_image_data_path, "r")) != NULL) {
    char buf[20];
    fread(&buf, sizeof(char), 16, fp); // skip
    test_images = (double **)malloc(sizeof(double *) * NTEST);
    unsigned char *pixels = (unsigned char *)malloc(sizeof(unsigned char) * NTEST * NPIXEL);
    fread(pixels, sizeof(unsigned char), NTEST * NPIXEL, fp);
    for (i = 0; i < NTEST; i++) {
      double *p = (double *)malloc(sizeof(double) * NPIXEL);
      for (j = 0; j < NPIXEL; j++) {
	p[j] = (double)pixels[i * NPIXEL + j] / 255.0;
      }
      test_images[i] = p;
    }
    free(pixels);
    fclose(fp);
  } else {
    printf("file (%s) open failed.\n", test_image_data_path);
  }

  test_mode(0);
}
