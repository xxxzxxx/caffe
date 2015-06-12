#include <limits>
#include <random>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  MAP_SMATRIX(eC, C, M, N);
  eC *= beta;
  if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
    MAP_CONST_SMATRIX(eA, A, M, K);
    MAP_CONST_SMATRIX(eB, B, K, N);
    eC.noalias() += alpha * (eA * eB);
  } else if (TransA == CblasNoTrans && TransB == CblasTrans) {
    MAP_CONST_SMATRIX(eA, A, M, K);
    MAP_CONST_SMATRIX(eB, B, N, K);
    eC.noalias() += alpha * (eA * eB.transpose());
  } else if (TransA == CblasTrans && TransB == CblasNoTrans) {
    MAP_CONST_SMATRIX(eA, A, K, M);
    MAP_CONST_SMATRIX(eB, B, K, N);
    eC.noalias() += alpha * (eA.transpose() * eB);
  } else {
    MAP_CONST_SMATRIX(eA, A, K, M);
    MAP_CONST_SMATRIX(eB, B, N, K);
    eC.noalias() += alpha * (eA.transpose() * eB.transpose());
  }
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  MAP_DMATRIX(eC, C, M, N);
  eC *= beta;
  if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
    MAP_CONST_DMATRIX(eA, A, M, K);
    MAP_CONST_DMATRIX(eB, B, K, N);
  eC.noalias() += alpha * (eA * eB);
  } else if (TransA == CblasNoTrans && TransB == CblasTrans) {
    MAP_CONST_DMATRIX(eA, A, M, K);
    MAP_CONST_DMATRIX(eB, B, N, K);
    eC.noalias() += alpha * (eA * eB.transpose());
  } else if (TransA == CblasTrans && TransB == CblasNoTrans) {
    MAP_CONST_DMATRIX(eA, A, K, M);
    MAP_CONST_DMATRIX(eB, B, K, N);
    eC.noalias() += alpha * (eA.transpose() * eB);
  } else {
    MAP_CONST_DMATRIX(eA, A, K, M);
    MAP_CONST_DMATRIX(eB, B, N, K);
    eC.noalias() += alpha * (eA.transpose() * eB.transpose());
  }
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  MAP_CONST_SMATRIX(eA, A, M, N);
  if (TransA == CblasNoTrans) {
    MAP_SVECTOR(eY, y, M);
    eY *= beta;
    MAP_CONST_SVECTOR(eX, x, N);
    eY.noalias() += alpha * (eA * eX);
  } else {
    MAP_SVECTOR(eY, y, N);
    eY *= beta;
    MAP_CONST_SVECTOR(eX, x, M);
    eY.noalias() += alpha * (eA.transpose() * eX);
  }
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  MAP_CONST_DMATRIX(eA, A, M, N);
  if (TransA == CblasNoTrans) {
    MAP_DVECTOR(eY, y, M);
    eY *= beta;
    MAP_CONST_DVECTOR(eX, x, N);
    eY.noalias() += alpha * (eA * eX);
  } else {
    MAP_DVECTOR(eY, y, N);
    eY *= beta;
    MAP_CONST_DVECTOR(eX, x, M);
    eY.noalias() += alpha * (eA.transpose() * eX);
  }
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  MAP_SVECTOR(eY, Y, N);
  MAP_CONST_SVECTOR(eX, X, N);
  eY = alpha * eX + eY;
}

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  MAP_DVECTOR(eY, Y, N);
  MAP_CONST_DVECTOR(eX, X, N);
  eY = alpha * eX + eY;
}

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  MAP_SVECTOR(eX, X, N);
  eX *= alpha;
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  MAP_DVECTOR(eX, X, N);
  eX *= alpha;
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
    MAP_SVECTOR(eY, Y, N);
  MAP_CONST_SVECTOR(eX, X, N);
  eY = alpha * eX + beta * eY;
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  MAP_DVECTOR(eY, Y, N);
  MAP_CONST_DVECTOR(eX, X, N);
  eY = alpha * eX + beta * eY;
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
#ifdef __ANDROID__
  return nextafter(b, std::numeric_limits<Dtype>::max());
#else
  return std::nextafter(b, std::numeric_limits<Dtype>::max());
#endif
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  std::uniform_real_distribution<>
      random_distribution(a, caffe_nextafter<Dtype>(b));
  std::function<Dtype()>
      variate_generator = bind(random_distribution, std::ref(*caffe_rng()));
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  std::normal_distribution<> random_distribution(a, sigma);
  std::function<Dtype()>
      variate_generator= bind(random_distribution, std::ref(*caffe_rng()));
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  std::bernoulli_distribution random_distribution(p);
  std::function<Dtype()>
      variate_generator = bind(random_distribution, std::ref(*caffe_rng()));
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  std::bernoulli_distribution random_distribution(p);
  std::function<Dtype() >
      variate_generator = bind(random_distribution, std::ref(*caffe_rng()));
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  int lx = (n + incx - 1) / incx;
  int ly = (n + incy - 1) / incy;
  MAP_CONST_SVECTOR_STRIDE(eX, x, lx, incx);
  MAP_CONST_SVECTOR_STRIDE(eY, y, ly, incy);
  return eX.dot(eY);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  int lx = (n + incx - 1) / incx;
  int ly = (n + incy - 1) / incy;
  MAP_CONST_DVECTOR_STRIDE(eX, x, lx, incx);
  MAP_CONST_DVECTOR_STRIDE(eY, y, ly, incy);
  return eX.dot(eY);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
int caffe_cpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(static_cast<uint32_t>(x[i]) ^
                               static_cast<uint32_t>(y[i]));
  }
  return dist;
}

template <>
int caffe_cpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcountl(static_cast<uint64_t>(x[i]) ^
                                static_cast<uint64_t>(y[i]));
  }
  return dist;
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  float *y = new float[n];
  memcpy(y, x, sizeof(float)*n);
  MAP_SVECTOR(eY, y, n);
  return eY.sum();
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  double *y = new double[n];
  memcpy(y, x, sizeof(double)*n);
  MAP_DVECTOR(eY, y, n);
  return eY.sum();
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  memcpy(y, x, sizeof(double)*n);
  MAP_SVECTOR(eY, y, n);
  eY *= alpha;
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  memcpy(y, x, sizeof(double)*n);
  MAP_DVECTOR(eY, y, n);
  eY *= alpha;
}

}  // namespace caffe
