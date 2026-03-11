/**
 * lu() and luComplex() — LU factorization with partial pivoting via LAPACK.
 *
 *   lu(data: Float64Array, m: number, n: number):
 *       {LU: Float64Array, ipiv: Int32Array}
 *
 *   luComplex(dataRe: Float64Array, dataIm: Float64Array, m: number, n: number):
 *       {LURe: Float64Array, LUIm: Float64Array, ipiv: Int32Array}
 *
 *     LU factorization of an m×n matrix stored in column-major order.
 *     Uses LAPACK dgetrf (real) / zgetrf (complex) with partial pivoting.
 *     Returns the packed LU matrix and 1-based pivot indices.
 */

#include "lapack_common.h"

// ── lu() ─────────────────────────────────────────────────────────────────────

Napi::Value Lu(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()) {
    Napi::TypeError::New(env,
      "lu: expected (Float64Array data, number m, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "lu: data must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m = info[1].As<Napi::Number>().Int32Value();
  int n = info[2].As<Napi::Number>().Int32Value();

  if (m <= 0 || n <= 0 || static_cast<int>(arr.ElementLength()) != m * n) {
    Napi::RangeError::New(env, "lu: data.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();
  int k = m < n ? m : n;

  // Copy input (dgetrf overwrites A in-place)
  std::vector<double> a(m * n);
  std::memcpy(a.data(), float64arr.Data(), m * n * sizeof(double));

  std::vector<int> ipiv(k);
  int info_val = 0;

  dgetrf_(&m, &n, a.data(), &m, ipiv.data(), &info_val);

  if (info_val < 0) {
    Napi::Error::New(env, "lu: illegal argument passed to dgetrf")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto LU_arr = Napi::Float64Array::New(env, static_cast<size_t>(m * n));
  std::memcpy(LU_arr.Data(), a.data(), m * n * sizeof(double));

  auto ipiv_arr = Napi::Int32Array::New(env, static_cast<size_t>(k));
  std::memcpy(ipiv_arr.Data(), ipiv.data(), k * sizeof(int));

  auto result = Napi::Object::New(env);
  result.Set("LU", LU_arr);
  result.Set("ipiv", ipiv_arr);
  return result;
}

// ── luComplex() ──────────────────────────────────────────────────────────────

Napi::Value LuComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsNumber()) {
    Napi::TypeError::New(env,
      "luComplex: expected (Float64Array dataRe, Float64Array dataIm, "
      "number m, number n)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();

  if (arrRe.TypedArrayType() != napi_float64_array ||
      arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "luComplex: dataRe and dataIm must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m = info[2].As<Napi::Number>().Int32Value();
  int n = info[3].As<Napi::Number>().Int32Value();

  if (m <= 0 || n <= 0 ||
      static_cast<int>(arrRe.ElementLength()) != m * n ||
      static_cast<int>(arrIm.ElementLength()) != m * n) {
    Napi::RangeError::New(env,
      "luComplex: dataRe.length and dataIm.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arrRe = info[0].As<Napi::Float64Array>();
  auto float64arrIm = info[1].As<Napi::Float64Array>();
  int k = m < n ? m : n;

  // Convert to interleaved complex format
  std::vector<lapack_complex_double> a(m * n);
  for (int i = 0; i < m * n; ++i) {
    a[i].real = float64arrRe[i];
    a[i].imag = float64arrIm[i];
  }

  std::vector<int> ipiv(k);
  int info_val = 0;

  zgetrf_(&m, &n, a.data(), &m, ipiv.data(), &info_val);

  if (info_val < 0) {
    Napi::Error::New(env, "luComplex: illegal argument passed to zgetrf")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto LURe_arr = Napi::Float64Array::New(env, static_cast<size_t>(m * n));
  auto LUIm_arr = Napi::Float64Array::New(env, static_cast<size_t>(m * n));
  for (int i = 0; i < m * n; ++i) {
    LURe_arr[i] = a[i].real;
    LUIm_arr[i] = a[i].imag;
  }

  auto ipiv_arr = Napi::Int32Array::New(env, static_cast<size_t>(k));
  std::memcpy(ipiv_arr.Data(), ipiv.data(), k * sizeof(int));

  auto result = Napi::Object::New(env);
  result.Set("LURe", LURe_arr);
  result.Set("LUIm", LUIm_arr);
  result.Set("ipiv", ipiv_arr);
  return result;
}
