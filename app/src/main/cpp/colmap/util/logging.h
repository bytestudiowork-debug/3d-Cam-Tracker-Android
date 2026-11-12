// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <stdexcept>
#include <type_traits>
namespace colmap {
template <typename T> T ThrowCheckNotNull(const char*, int, const char*, T value) {
  if (value == nullptr) throw std::invalid_argument("Null pointer check failed");
  return value;
}
inline void ThrowCheckFailed(const char* expression) { throw std::invalid_argument(expression); }
}
#define THROW_CHECK(c) do { if (!(c)) colmap::ThrowCheckFailed(#c); } while (false)
#define THROW_CHECK_EQ(a,b) do { if (!((a)==(b))) colmap::ThrowCheckFailed(#a " == " #b); } while (false)
#define THROW_CHECK_NE(a,b) do { if (!((a)!=(b))) colmap::ThrowCheckFailed(#a " != " #b); } while (false)
#define THROW_CHECK_LE(a,b) do { if (!((a)<=(b))) colmap::ThrowCheckFailed(#a " <= " #b); } while (false)
#define THROW_CHECK_LT(a,b) do { if (!((a)<(b))) colmap::ThrowCheckFailed(#a " < " #b); } while (false)
#define THROW_CHECK_GE(a,b) do { if (!((a)>=(b))) colmap::ThrowCheckFailed(#a " >= " #b); } while (false)
#define THROW_CHECK_GT(a,b) do { if (!((a)>(b))) colmap::ThrowCheckFailed(#a " > " #b); } while (false)
#define THROW_CHECK_NOTNULL(v) colmap::ThrowCheckNotNull(__FILE__, __LINE__, #v, (v))
