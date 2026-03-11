/**
 * matmul() — Real matrix-matrix multiplication via BLAS dgemm.
 *
 *   matmul(A: Float64Array, m: number, k: number,
 *          B: Float64Array, n: number): Float64Array
 *
 *     Computes C = A * B where:
 *       A is an m×k matrix stored in column-major order
 *       B is a  k×n matrix stored in column-major order
 *       C is an m×n matrix returned in column-major order
 *
 *     Uses BLAS dgemm for high-performance computation.
 *     Equivalent to MATLAB: C = A * B
 */

#include "lapack_common.h"

// ── matmul() ──────────────────────────────────────────────────────────────────

Napi::Value Matmul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()
      || !info[3].IsTypedArray()
      || !info[4].IsNumber()) {
    Napi::TypeError::New(env,
      "matmul: expected (Float64Array A, number m, number k, Float64Array B, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrA = info[0].As<Napi::TypedArray>();
  auto arrB = info[3].As<Napi::TypedArray>();

  if (arrA.TypedArrayType() != napi_float64_array ||
      arrB.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "matmul: A and B must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m = info[1].As<Napi::Number>().Int32Value(); // rows of A and C
  int k = info[2].As<Napi::Number>().Int32Value(); // cols of A, rows of B
  int n = info[4].As<Napi::Number>().Int32Value(); // cols of B and C

  if (m < 0 || k < 0 || n < 0) {
    Napi::RangeError::New(env, "matmul: m, k, n must be non-negative")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Handle empty-dimension multiply without calling dgemm.
  // BLAS requires ldb >= max(1, k), so k=0 would be invalid (ldb=0 < 1).
  if (m == 0 || k == 0 || n == 0) {
    auto result = Napi::Float64Array::New(env, static_cast<size_t>(m * n));
    // Zero-initialized by default in V8.
    return result;
  }
  if (static_cast<int>(arrA.ElementLength()) != m * k) {
    Napi::RangeError::New(env, "matmul: A.length must equal m*k")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (static_cast<int>(arrB.ElementLength()) != k * n) {
    Napi::RangeError::New(env, "matmul: B.length must equal k*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64A = info[0].As<Napi::Float64Array>();
  auto float64B = info[3].As<Napi::Float64Array>();

  // ── Compute C = A * B via dgemm ───────────────────────────────────────────
  // dgemm computes: C = alpha * op(A) * op(B) + beta * C
  // With transa='N', transb='N', alpha=1, beta=0 this gives C = A * B.
  char transa = 'N';
  char transb = 'N';
  double alpha = 1.0;
  double beta  = 0.0;

  // dgemm args: lda = leading dim of A = m (column-major)
  //             ldb = leading dim of B = k
  //             ldc = leading dim of C = m
  int lda = m;
  int ldb = k;
  int ldc = m;

  std::vector<double> c(m * n);

  dgemm_(&transa, &transb,
         &m, &n, &k,
         &alpha, float64A.Data(), &lda,
                 float64B.Data(), &ldb,
         &beta,  c.data(),        &ldc);

  // ── Return result as a new Float64Array ───────────────────────────────────
  auto result = Napi::Float64Array::New(env, static_cast<size_t>(m * n));
  std::memcpy(result.Data(), c.data(), m * n * sizeof(double));
  return result;
}
