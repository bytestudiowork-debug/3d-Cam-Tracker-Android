




/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_MATHOP_H
#define VL_MATHOP_H

#include "generic.h"
#include <math.h>
#include <float.h>


#define VL_E 2.718281828459045


#define VL_LOG_OF_2 0.693147180559945


#define VL_PI 3.141592653589793








#define VL_EPSILON_F 1.19209290E-07F







#define VL_EPSILON_D 2.220446049250313e-16









static union { vl_uint32 raw ; float value ; }
  const vl_nan_f =
    { 0x7FC00000UL } ;


static union { vl_uint32 raw ; float value ; }
  const vl_infinity_f =
    { 0x7F800000UL } ;


static union { vl_uint64 raw ; double value ; }
  const vl_nan_d =
#ifdef VL_COMPILER_MSC
    { 0x7FF8000000000000ui64 } ;
#else
    { 0x7FF8000000000000ULL } ;
#endif


static union { vl_uint64 raw ; double value ; }
  const vl_infinity_d =
#ifdef VL_COMPILER_MSC
    { 0x7FF0000000000000ui64 } ;
#else
    { 0x7FF0000000000000ULL } ;
#endif


#define VL_NAN_F (vl_nan_f.value)


#define VL_INFINITY_F (vl_infinity_f.value)


#define VL_NAN_D (vl_nan_d.value)


#define VL_INFINITY_D (vl_infinity_d.value)
















VL_INLINE float
vl_mod_2pi_f (float x)
{
  while (x > (float)(2 * VL_PI)) x -= (float) (2 * VL_PI) ;
  while (x < 0.0F) x += (float) (2 * VL_PI);
  return x ;
}





VL_INLINE double
vl_mod_2pi_d (double x)
{
  while (x > 2.0 * VL_PI) x -= 2 * VL_PI ;
  while (x < 0.0) x += 2 * VL_PI ;
  return x ;
}






VL_INLINE long int
vl_floor_f (float x)
{
  long int xi = (long int) x ;
  if (x >= 0 || (float) xi == x) return xi ;
  else return xi - 1 ;
}





VL_INLINE long int
vl_floor_d (double x)
{
  long int xi = (long int) x ;
  if (x >= 0 || (double) xi == x) return xi ;
  else return xi - 1 ;
}






VL_INLINE long int
vl_ceil_f (float x)
{
#ifdef VL_COMPILER_GNUC
  return (long int) __builtin_ceilf(x) ;
#else
  return (long int) ceilf(x) ;
#endif
}





VL_INLINE long int
vl_ceil_d (double x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_ceil(x) ;
#else
  return (long int) ceil(x) ;
#endif
}







VL_INLINE long int
vl_round_f (float x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_lroundf(x) ;
#elif VL_COMPILER_MSC
  if (x >= 0.0F) {
    return vl_floor_f(x + 0.5F) ;
  } else {
    return vl_ceil_f(x - 0.5F) ;
  }
#else
  return lroundf(x) ;
#endif
}







VL_INLINE long int
vl_round_d (double x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_lround(x) ;
#elif VL_COMPILER_MSC
  if (x >= 0.0) {
    return vl_floor_d(x + 0.5) ;
  } else {
    return vl_ceil_d(x - 0.5) ;
  }
#else
  return lround(x) ;
#endif
}






VL_INLINE float
vl_abs_f (float x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_fabsf (x) ;
#else
  return fabsf(x) ;
#endif
}





VL_INLINE double
vl_abs_d (double x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_fabs (x) ;
#else
  return fabs(x) ;
#endif
}






VL_INLINE double
vl_log2_d (double x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_log2(x) ;
#elif VL_COMPILER_MSC
  return log(x) / 0.693147180559945 ;
#else
  return log2(x) ;
#endif
}


VL_INLINE float
vl_log2_f (float x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_log2f (x) ;
#elif VL_COMPILER_MSC
  return logf(x) / 0.6931472F ;
#else
  return log2(x) ;
#endif
}






VL_INLINE double
vl_sqrt_d (double x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_sqrt(x) ;
#else
  return sqrt(x) ;
#endif
}


VL_INLINE float
vl_sqrt_f (float x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_sqrtf(x) ;
#else
  return sqrtf(x) ;
#endif
}






VL_INLINE vl_bool
vl_is_nan_f (float x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_isnan (x) ;
#elif VL_COMPILER_MSC
  return _isnan(x) ;
#else
  return isnan(x) ;
#endif
}


VL_INLINE vl_bool
vl_is_nan_d (double x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_isnan (x) ;
#elif VL_COMPILER_MSC
  return _isnan(x) ;
#else
  return isnan(x) ;
#endif
}





VL_INLINE vl_bool
vl_is_inf_f (float x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_isinf (x) ;
#elif VL_COMPILER_MSC
  return ! _finite(x) ;
#else
  return isinf(x) ;
#endif
}


VL_INLINE vl_bool
vl_is_inf_d (double x)
{
#ifdef VL_COMPILER_GNUC
  return __builtin_isinf (x) ;
#elif VL_COMPILER_MSC
  return ! _finite(x) ;
#else
  return isinf(x) ;
#endif
}





































VL_INLINE float
vl_fast_atan2_f (float y, float x)
{
  float angle, r ;
  float const c3 = 0.1821F ;
  float const c1 = 0.9675F ;
  float abs_y    = vl_abs_f (y) + VL_EPSILON_F ;

  if (x >= 0) {
    r = (x - abs_y) / (x + abs_y) ;
    angle = (float) (VL_PI / 4) ;
  } else {
    r = (x + abs_y) / (abs_y - x) ;
    angle = (float) (3 * VL_PI / 4) ;
  }
  angle += (c3*r*r - c1) * r ;
  return (y < 0) ? - angle : angle ;
}





VL_INLINE double
vl_fast_atan2_d (double y, double x)
{
  double angle, r ;
  double const c3 = 0.1821 ;
  double const c1 = 0.9675 ;
  double abs_y = vl_abs_d (y) + VL_EPSILON_D ;

  if (x >= 0) {
    r = (x - abs_y) / (x + abs_y) ;
    angle = VL_PI / 4 ;
  } else {
    r = (x + abs_y) / (abs_y - x) ;
    angle = 3 * VL_PI / 4 ;
  }
  angle += (c3*r*r - c1) * r ;
  return (y < 0) ? - angle : angle ;
}































VL_INLINE float
vl_fast_resqrt_f (float x)
{

  union {
    float x ;
    vl_int32  i ;
  } u ;

  float xhalf = (float) 0.5 * x ;


  u.x = x ;


  u.i = 0x5f3759df - (u.i >> 1);



  u.x = u.x * ( (float) 1.5  - xhalf*u.x*u.x) ;
  u.x = u.x * ( (float) 1.5  - xhalf*u.x*u.x) ;
  return u.x ;
}





VL_INLINE double
vl_fast_resqrt_d (double x)
{

  union {
    double x ;
    vl_int64  i ;
  } u ;

  double xhalf = (double) 0.5 * x ;


  u.x = x ;


#ifdef VL_COMPILER_MSC
  u.i = 0x5fe6ec85e7de30dai64 - (u.i >> 1) ;
#else
  u.i = 0x5fe6ec85e7de30daLL - (u.i >> 1) ;
#endif


  u.x = u.x * ( (double) 1.5  - xhalf*u.x*u.x) ;
  u.x = u.x * ( (double) 1.5  - xhalf*u.x*u.x) ;
  return u.x ;
}











VL_INLINE float
vl_fast_sqrt_f (float x)
{
  return (x < 1e-8) ? 0 : x * vl_fast_resqrt_f (x) ;
}





VL_INLINE double
vl_fast_sqrt_d (float x)
{
  return (x < 1e-8) ? 0 : x * vl_fast_resqrt_d (x) ;
}






VL_INLINE vl_uint64 vl_fast_sqrt_ui64 (vl_uint64 x) ;



VL_INLINE vl_uint32 vl_fast_sqrt_ui32 (vl_uint32 x) ;



VL_INLINE vl_uint16 vl_fast_sqrt_ui16 (vl_uint16 x) ;



VL_INLINE vl_uint8  vl_fast_sqrt_ui8  (vl_uint8  x) ;

#define VL_FAST_SQRT_UI(T,SFX)                                       \
VL_INLINE T                                                          \
vl_fast_sqrt_ ## SFX (T x)                                           \
{                                                                    \
  T y = 0 ;                                                          \
  T tmp = 0 ;                                                        \
  int twice_k ;                                                      \
  for (twice_k = 8 * sizeof(T) - 2 ;                                 \
       twice_k >= 0 ; twice_k -= 2) {                                \
    y <<= 1 ;                                                        \
    tmp = (2*y + 1) << twice_k ;                                     \
    if (x >= tmp) {                                                  \
      x -= tmp ;                                                     \
      y += 1 ;                                                       \
    }                                                                \
  }                                                                  \
  return y ;                                                         \
}

VL_FAST_SQRT_UI(vl_uint64,ui64)
VL_FAST_SQRT_UI(vl_uint32,ui32)
VL_FAST_SQRT_UI(vl_uint16,ui16)
VL_FAST_SQRT_UI(vl_uint8,ui8)








typedef float (*VlFloatVectorComparisonFunction)(vl_size dimension, float const * X, float const * Y) ;




typedef double (*VlDoubleVectorComparisonFunction)(vl_size dimension, double const * X, double const * Y) ;




typedef float (*VlFloatVector3ComparisonFunction)(vl_size dimension, float const * X, float const * Y, float const * Z) ;




typedef double (*VlDoubleVector3ComparisonFunction)(vl_size dimension, double const * X, double const * Y, double const * Z) ;


enum _VlVectorComparisonType {
  VlDistanceL1,
  VlDistanceL2,
  VlDistanceChi2,
  VlDistanceHellinger,
  VlDistanceJS,
  VlDistanceMahalanobis,
  VlKernelL1,
  VlKernelL2,
  VlKernelChi2,
  VlKernelHellinger,
  VlKernelJS
} ;


typedef enum _VlVectorComparisonType VlVectorComparisonType ;






VL_INLINE char const *
vl_get_vector_comparison_type_name (int type)
{
  switch (type) {
    case VlDistanceL1   : return "l1" ;
    case VlDistanceL2   : return "l2" ;
    case VlDistanceChi2 : return "chi2" ;
    case VlDistanceMahalanobis  : return "mahalanobis" ;
    case VlKernelL1     : return "kl1" ;
    case VlKernelL2     : return "kl2" ;
    case VlKernelChi2   : return "kchi2" ;
    default: return NULL ;
  }
}

VL_EXPORT VlFloatVectorComparisonFunction
vl_get_vector_comparison_function_f (VlVectorComparisonType type) ;

VL_EXPORT VlDoubleVectorComparisonFunction
vl_get_vector_comparison_function_d (VlVectorComparisonType type) ;

VL_EXPORT VlFloatVector3ComparisonFunction
vl_get_vector_3_comparison_function_f (VlVectorComparisonType type) ;

VL_EXPORT VlDoubleVector3ComparisonFunction
vl_get_vector_3_comparison_function_d (VlVectorComparisonType type) ;


VL_EXPORT void
vl_eval_vector_comparison_on_all_pairs_f (float * result, vl_size dimension,
                                          float const * X, vl_size numDataX,
                                          float const * Y, vl_size numDataY,
                                          VlFloatVectorComparisonFunction function) ;

VL_EXPORT void
vl_eval_vector_comparison_on_all_pairs_d (double * result, vl_size dimension,
                                          double const * X, vl_size numDataX,
                                          double const * Y, vl_size numDataY,
                                          VlDoubleVectorComparisonFunction function) ;





VL_EXPORT void
vl_svd2 (double* S, double *U, double *V, double const *M) ;

VL_EXPORT void
vl_lapack_dlasv2 (double *smin,
                  double *smax,
                  double *sv,
                  double *cv,
                  double *su,
                  double *cu,
                  double f,
                  double g,
                  double h) ;


VL_EXPORT int
vl_solve_linear_system_3 (double * x, double const * A, double const *b) ;

VL_EXPORT int
vl_solve_linear_system_2 (double * x, double const * A, double const *b) ;

VL_EXPORT int
vl_gaussian_elimination (double * A, vl_size numRows, vl_size numColumns) ;


#endif
