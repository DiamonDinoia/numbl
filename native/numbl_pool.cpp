/**
 * Pool allocator for typed arrays backed by externally-allocated memory.
 *
 * Uses aligned_alloc with 64-byte alignment (cache-line) and wraps the
 * result in napi_external_arraybuffer so V8 skips its own bookkeeping.
 * The finalize callback automatically frees memory when the array is GC'd.
 */

#include "numbl_addon_common.h"
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <malloc.h>
#define pool_aligned_alloc(align, size) _aligned_malloc(size, align)
#define pool_aligned_free(ptr) _aligned_free(ptr)
#else
#define pool_aligned_alloc(align, size) aligned_alloc(align, size)
#define pool_aligned_free(ptr) free(ptr)
#endif

bool pool_enabled = false;

static size_t align_up(size_t size, size_t align) {
  return (size + align - 1) & ~(align - 1);
}

static void PoolFinalizer(Napi::Env /*env*/, void* data) {
  pool_aligned_free(data);
}

Napi::Value PoolInit(const Napi::CallbackInfo& info) {
  pool_enabled = info[0].As<Napi::Boolean>().Value();
  return info.Env().Undefined();
}

Napi::Value PoolEnabled(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), pool_enabled);
}

// ── Float64 ─────────────────────────────────────────────────────────────────

Napi::Value PoolAllocFloat64(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  size_t length = info[0].As<Napi::Number>().Uint32Value();
  size_t bytes = align_up(length * sizeof(double), 64);
  if (bytes == 0) bytes = 64;
  void* ptr = pool_aligned_alloc(64, bytes);
  if (!ptr) return Napi::Float64Array::New(env, length);
  std::memset(ptr, 0, bytes);
  auto ab = Napi::ArrayBuffer::New(env, ptr, length * sizeof(double), PoolFinalizer);
  return Napi::Float64Array::New(env, length, ab, 0);
}

Napi::Value PoolAllocFloat64From(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto source = info[0].As<Napi::Float64Array>();
  size_t length = source.ElementLength();
  size_t bytes = align_up(length * sizeof(double), 64);
  if (bytes == 0) bytes = 64;
  void* ptr = pool_aligned_alloc(64, bytes);
  if (!ptr) {
    auto arr = Napi::Float64Array::New(env, length);
    std::memcpy(arr.Data(), source.Data(), length * sizeof(double));
    return arr;
  }
  std::memcpy(ptr, source.Data(), length * sizeof(double));
  auto ab = Napi::ArrayBuffer::New(env, ptr, length * sizeof(double), PoolFinalizer);
  return Napi::Float64Array::New(env, length, ab, 0);
}

