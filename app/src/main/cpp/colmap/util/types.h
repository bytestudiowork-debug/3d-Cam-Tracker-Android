// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstddef>
#include <Eigen/Core>
namespace Eigen {
using Matrix3x4d = Matrix<double, 3, 4>;
using Matrix2x3d = Matrix<double, 2, 3>;
}
namespace colmap {
template <typename T> class span {
 public:
  span(T* data, size_t size) noexcept : data_(data), size_(size) {}
  T& operator[](size_t i) noexcept { return data_[i]; }
  const T& operator[](size_t i) const noexcept { return data_[i]; }
  size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  T* begin() noexcept { return data_; }
  T* end() noexcept { return data_ + size_; }
  const T* begin() const noexcept { return data_; }
  const T* end() const noexcept { return data_ + size_; }
 private: T* data_; size_t size_;
};