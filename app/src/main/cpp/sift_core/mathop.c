




/*
Copyright (C) 2014 Andrea Vedaldi.
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/
























































































































































































































































#ifndef VL_MATHOP_INSTANTIATING

#include "mathop.h"
#include "mathop_sse2.h"
 #include "mathop_avx.h"
#include <math.h>

#undef FLT
#define FLT VL_TYPE_FLOAT
#define VL_MATHOP_INSTANTIATING
#include "mathop.c"

#undef FLT
#define FLT VL_TYPE_DOUBLE
#define VL_MATHOP_INSTANTIATING
#include "mathop.c"
#endif


#ifdef VL_MATHOP_INSTANTIATING
#include "float.h"

#undef COMPARISONFUNCTION_TYPE
#undef COMPARISONFUNCTION3_TYPE
#if (FLT == VL_TYPE_FLOAT)
#  define COMPARISONFUNCTION_TYPE VlFloatVectorComparisonFunction
#  define COMPARISONFUNCTION3_TYPE VlFloatVector3ComparisonFunction
#else
#  define COMPARISONFUNCTION_TYPE VlDoubleVectorComparisonFunction
#  define COMPARISONFUNCTION3_TYPE VlDoubleVector3ComparisonFunction
#endif



VL_EXPORT T
VL_XCAT(_vl_distance_l2_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T d = *X++ - *Y++ ;
    acc += d * d ;
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_l1_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T d = *X++ - *Y++ ;
    acc += VL_MAX(d, -d) ;
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_chi2_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    T delta = a - b ;
    T denom = (a + b) ;
    T numer = delta * delta ;
    if (denom) {
      T ratio = numer / denom ;
      acc += ratio ;
    }
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_hellinger_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
#if (FLT == VL_TYPE_FLOAT)
    acc += a + b - 2.0 * sqrtf (a*b) ;
#else
    acc += a + b - 2.0 * sqrt (a*b) ;
#endif
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_js_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T x = *X++ ;
    T y = *Y++ ;
    if (x) acc += x - x * VL_XCAT(vl_log2_,SFX)(1 + y/x) ;
    if (y) acc += y - y * VL_XCAT(vl_log2_,SFX)(1 + x/y) ;
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_l2_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    acc += a * b ;
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_l1_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    T a_ = VL_XCAT(vl_abs_, SFX) (a) ;
    T b_ = VL_XCAT(vl_abs_, SFX) (b) ;
    acc += a_ + b_ - VL_XCAT(vl_abs_, SFX) (a - b) ;
  }
  return acc / ((T)2) ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_chi2_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    T denom = (a + b) ;
    if (denom) {
      T numer = 2 * a * b ;
      T ratio = numer / denom ;
      acc += ratio ;
    }
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_hellinger_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
#if (FLT == VL_TYPE_FLOAT)
    acc += sqrtf (a*b) ;
#else
    acc += sqrt (a*b) ;
#endif
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_js_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T x = *X++ ;
    T y = *Y++ ;
    if (x) acc += x * VL_XCAT(vl_log2_,SFX)(1 + y/x) ;
    if (y) acc += y * VL_XCAT(vl_log2_,SFX)(1 + x/y) ;
  }
  return (T)0.5 * acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_mahalanobis_sq_, SFX)
(vl_size dimension, T const * X, T const * MU, T const * S)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T d = *X++ - *MU++ ;
    acc += d * d * (*S++) ;
  }
  return acc ;
}



VL_EXPORT COMPARISONFUNCTION_TYPE
VL_XCAT(vl_get_vector_comparison_function_, SFX)(VlVectorComparisonType type)
{
  COMPARISONFUNCTION_TYPE function = 0 ;
  switch (type) {
    case VlDistanceL2        : function = VL_XCAT(_vl_distance_l2_,             SFX) ; break ;
    case VlDistanceL1        : function = VL_XCAT(_vl_distance_l1_,             SFX) ; break ;
    case VlDistanceChi2      : function = VL_XCAT(_vl_distance_chi2_,           SFX) ; break ;
    case VlDistanceHellinger : function = VL_XCAT(_vl_distance_hellinger_,      SFX) ; break ;
    case VlDistanceJS        : function = VL_XCAT(_vl_distance_js_,             SFX) ; break ;
    case VlKernelL2          : function = VL_XCAT(_vl_kernel_l2_,               SFX) ; break ;
    case VlKernelL1          : function = VL_XCAT(_vl_kernel_l1_,               SFX) ; break ;
    case VlKernelChi2        : function = VL_XCAT(_vl_kernel_chi2_,             SFX) ; break ;
    case VlKernelHellinger   : function = VL_XCAT(_vl_kernel_hellinger_,        SFX) ; break ;
    case VlKernelJS          : function = VL_XCAT(_vl_kernel_js_,               SFX) ; break ;
    default: abort() ;
  }

#ifndef VL_DISABLE_SSE2

  if (vl_cpu_has_sse2() && vl_get_simd_enabled()) {
    switch (type) {
      case VlDistanceL2    : function = VL_XCAT(_vl_distance_l2_sse2_,             SFX) ; break ;
      case VlDistanceL1    : function = VL_XCAT(_vl_distance_l1_sse2_,             SFX) ; break ;
      case VlDistanceChi2  : function = VL_XCAT(_vl_distance_chi2_sse2_,           SFX) ; break ;
      case VlKernelL2      : function = VL_XCAT(_vl_kernel_l2_sse2_,               SFX) ; break ;
      case VlKernelL1      : function = VL_XCAT(_vl_kernel_l1_sse2_,               SFX) ; break ;
      case VlKernelChi2    : function = VL_XCAT(_vl_kernel_chi2_sse2_,             SFX) ; break ;
      default: break ;
    }
  }
#endif

#ifndef VL_DISABLE_AVX

  if (vl_cpu_has_avx() && vl_get_simd_enabled()) {
    switch (type) {
      case VlDistanceL2    : function = VL_XCAT(_vl_distance_l2_avx_,             SFX) ; break ;
      default: break ;
    }
  }
#endif

  return function ;
}



VL_EXPORT COMPARISONFUNCTION3_TYPE
VL_XCAT(vl_get_vector_3_comparison_function_, SFX)(VlVectorComparisonType type)
{
  COMPARISONFUNCTION3_TYPE function = 0 ;
  switch (type) {
    case VlDistanceMahalanobis : function = VL_XCAT(_vl_distance_mahalanobis_sq_, SFX) ; break ;
    default: abort() ;
  }

#ifndef VL_DISABLE_SSE2

  if (vl_cpu_has_sse2() && vl_get_simd_enabled()) {
    switch (type) {
      case VlDistanceMahalanobis : function = VL_XCAT(_vl_distance_mahalanobis_sq_sse2_, SFX) ; break ;
      default: break ;
    }
  }
#endif

#ifndef VL_DISABLE_AVX

  if (vl_cpu_has_avx() && vl_get_simd_enabled()) {
    switch (type) {
      case VlDistanceMahalanobis : function = VL_XCAT(_vl_distance_mahalanobis_sq_avx_, SFX) ; break ;
      default: break ;
    }
  }
#endif

  return function ;
}



VL_EXPORT void
VL_XCAT(vl_eval_vector_comparison_on_all_pairs_, SFX)
(T * result, vl_size dimension,
 T const * X, vl_size numDataX,
 T const * Y, vl_size numDataY,
 COMPARISONFUNCTION_TYPE function)
{
  vl_uindex xi ;
  vl_uindex yi ;

  if (dimension == 0) return ;
  if (numDataX == 0) return ;
  assert (X) ;

  if (Y) {
    if (numDataY == 0) return ;
    for (yi = 0 ; yi < numDataY ; ++ yi) {
      for (xi = 0 ; xi < numDataX ; ++ xi) {
        *result++ = (*function)(dimension, X, Y) ;
        X += dimension ;
      }
      X -= dimension * numDataX ;
      Y += dimension ;
    }
  } else {
    T * resultTransp = result ;
    Y = X ;
    for (yi = 0 ; yi < numDataX ; ++ yi) {
      for (xi = 0 ; xi <= yi ; ++ xi) {
        T z = (*function)(dimension, X, Y) ;
        X += dimension ;
        *result       = z ;
        *resultTransp = z ;
        result        += 1 ;
        resultTransp  += numDataX ;
      }
      X -= dimension * (yi + 1) ;
      Y += dimension ;
      result       += numDataX - (yi + 1) ;
      resultTransp += 1        - (yi + 1) * numDataX ;
    }
  }
}


#endif






#ifndef VL_MATHOP_INSTANTIATING

































void
vl_svd2 (double* S, double *U, double *V, double const *M)
{
  double m11 = M[0] ;
  double m21 = M[1] ;
  double m12 = M[2] ;
  double m22 = M[3] ;
  double cu1 = m11 ;
  double su1 = m21 ;
  double norm = sqrt(cu1*cu1 + su1*su1) ;
  double cu2, su2, cv2, sv2 ;
  double f, g, h ;
  double smin, smax ;
  cu1 /= norm ;
  su1 /= norm ;

  f = cu1 * m11 + su1 * m21 ;
  g = cu1 * m12 + su1 * m22 ;
  h = - su1 * m12 + cu1 * m22 ;

  vl_lapack_dlasv2 (&smin, &smax,
                    &sv2, &cv2,
                    &su2, &cu2,
                    f, g, h) ;

  assert(S) ;
  S[0] = smax ;
  S[1] = 0 ;
  S[2] = 0 ;
  S[3] = smin ;

  if (U) {
    U[0] = cu2*cu1 - su2*su1 ;
    U[1] = su2*cu1 + cu2*su1 ;
    U[2] = - cu2*su1 - su2*cu1 ;
    U[3] = - su2*su1 + cu2*cu1 ;
  }
  if (V) {
    V[0] = cv2 ;
    V[1] = sv2 ;
    V[2] = - sv2 ;
    V[3] = cv2 ;
  }
}

























#define isign(i) ((i)<0 ? (-1) : (+1))
#define sign(x) ((x)<0.0 ? (-1) : (+1))

void
vl_lapack_dlasv2 (double *smin,
                  double *smax,
                  double *sv,
                  double *cv,
                  double *su,
                  double *cu,
                  double f,
                  double g,
                  double h)
{
  double svt, cvt, sut, cut;
  double ft = f, gt = g, ht = h;
  double fa = fabs(f), ga = fabs(g), ha = fabs(h);
  int pmax = 1 ;
  int swap = 0 ;
  int glarge = 0 ;
  int tsign ;
  double fmh ;
  double d ;
  double dd ;
  double q ;
  double qq ;
  double s ;
  double ss ;
  double spq ;
  double dpq ;
  double a ;
  double tmp ;
  double tt;


  if (fa < ha) {
    pmax = 3 ;
    tmp =ft ; ft = ht ; ht = tmp ;
    tmp =fa ; fa = ha ; ha = tmp ;
    swap = 1 ;
  }

  if (ga == 0.0) {
    *smin = ha ;
    *smax = fa ;

    cut = 1.0 ; sut = 0.0 ;
    cvt = 1.0 ; svt = 0.0 ;
  }
  else {
    if (ga > fa) {
      pmax = 2 ;
      if ((fa / ga) < VL_EPSILON_D) {
        glarge = 1 ;
        *smax = ga ;
        if (ha > 1.0) {
          *smin = fa / (ga / ha) ;
        } else {
          *smin = (fa / ga) * ha ;
        }
        cut = 1.0 ; sut = ht / gt ;
        cvt = 1.0 ; svt = ft / gt ;
      }
    }

    if (glarge == 0) {
      fmh = fa - ha ;
      if (fmh == fa) {
        d = 1.0 ;
      } else {
        d = fmh / fa ;
      }
      q = gt / ft ;
      s = 2.0 - d ;
      dd = d*d ;
      qq = q*q ;
      ss = s*s ;
      spq = sqrt(ss + qq) ;
      if (d == 0.0) {
        dpq = fabs(q) ;
      } else {
        dpq = sqrt(dd + qq) ;
      }
      a = 0.5 * (spq + dpq) ;
      *smin = ha / a;
      *smax = fa * a;
      if (qq==0.0) {
        if (d==0.0) {
          tmp = sign(ft)*2*sign(gt);
        }
        else {
          tmp = gt/(sign(ft)*fmh) + q/s;
        }
      } else {
        tmp = (q/(spq + s) + q/(dpq + d))*(1.0 + a);
      }

      tt = sqrt(tmp*tmp + 4.0) ;
      cvt = 2.0 / tt ;
      svt = tmp / tt ;
      cut = (cvt + svt*q) / a ;
      sut = (ht / ft) * svt / a ;
    }
  }
  if (swap == 1) {
    *cu = svt ; *su = cvt ;
    *cv = sut ; *sv = cut ;
  } else {
    *cu = cut ; *su = sut ;
    *cv = cvt ; *sv = svt ;
  }

  if (pmax==1) { tsign = sign(*cv) * sign(*cu) * sign(f) ; }
  if (pmax==2) { tsign = sign(*sv) * sign(*cu) * sign(g) ; }
  if (pmax==3) { tsign = sign(*sv) * sign(*su) * sign(h) ; }
  *smax = isign(tsign) * (*smax);
  *smin = isign(tsign * sign(f) * sign(h)) * (*smin) ;
}











VL_EXPORT int
vl_solve_linear_system_3 (double * x, double const * A, double const *b)
{
  int err ;
  double M[3*4] ;
  M[0] = A[0] ;
  M[1] = A[1] ;
  M[2] = A[2] ;
  M[3] = A[3] ;
  M[4] = A[4] ;
  M[5] = A[5] ;
  M[6] = A[6] ;
  M[7] = A[7] ;
  M[8] = A[8] ;
  M[9] = b[0] ;
  M[10] = b[1] ;
  M[11] = b[2] ;
  err = vl_gaussian_elimination(M,3,4) ;
  x[0] = M[9] ;
  x[1] = M[10] ;
  x[2] = M[11] ;
  return err ;
}










VL_EXPORT int
vl_solve_linear_system_2 (double * x, double const * A, double const *b)
{
  int err ;
  double M[2*3] ;
  M[0] = A[0] ;
  M[1] = A[1] ;
  M[2] = A[2] ;
  M[3] = A[3] ;
  M[4] = b[0];
  M[5] = b[1] ;
  err = vl_gaussian_elimination(M,2,3) ;
  x[0] = M[4] ;
  x[1] = M[5] ;
  return err ;
}


















VL_EXPORT vl_bool
vl_gaussian_elimination (double * A, vl_size numRows, vl_size numColumns)
{
  vl_index i, j, ii, jj ;
  assert(A) ;
  assert(numRows <= numColumns) ;

#define Aat(i,j) A[(i) + (j)*numRows]


  for(j = 0 ; j < (signed)numRows ; ++j) {
    double maxa = 0 ;
    double maxabsa = 0 ;
    vl_index maxi = -1 ;
    double tmp ;

#if 0
    {
      vl_index iii, jjj ;
      for (iii = 0 ; iii < 2 ; ++iii) {
        for (jjj = 0 ; jjj < 3 ; ++jjj) {
          VL_PRINTF("%5.2g ", Aat(iii,jjj)) ;

        }
        VL_PRINTF("\n") ;
      }
      VL_PRINTF("\n") ;
    }
#endif


    for (i = j ; i < (signed)numRows ; ++i) {
      double a = Aat(i,j) ;
      double absa = vl_abs_d (a) ;
      if (absa > maxabsa) {
        maxa = a ;
        maxabsa = absa ;
        maxi = i ;
      }
    }
    i = maxi ;


    if (maxabsa < 1e-10) return VL_ERR_OVERFLOW ;


    for(jj = j ; jj < (signed)numColumns ; ++jj) {
      tmp = Aat(i,jj) ; Aat(i,jj) = Aat(j,jj) ; Aat(j,jj) = tmp ;
      Aat(j,jj) /= maxa ;
    }

#if 0
    {
      vl_index iii, jjj ;
      VL_PRINTF("after swap %d %d\n", j, i);
      for (iii = 0 ; iii < 2 ; ++iii) {
        for (jjj = 0 ; jjj < 3 ; ++jjj) {
          VL_PRINTF("%5.2g ", Aat(iii,jjj)) ;

        }
        VL_PRINTF("\n") ;
      }
      VL_PRINTF("\n") ;
    }
#endif


    for (ii = j+1 ; ii < (signed)numRows ; ++ii) {
      double x = Aat(ii,j) ;
      for (jj = j ; jj < (signed)numColumns ; ++jj) {
        Aat(ii,jj) -= x * Aat(j,jj) ;
      }
    }

#if 0
    {
      VL_PRINTF("after elimination\n");

      vl_index iii, jjj ;
      for (iii = 0 ; iii < 2 ; ++iii) {
        for (jjj = 0 ; jjj < 3 ; ++jjj) {
          VL_PRINTF("%5.2g ", Aat(iii,jjj)) ;

        }
        VL_PRINTF("\n") ;
      }
      VL_PRINTF("\n") ;
    }
#endif

  }


  for (i = numRows - 1 ; i > 0 ; --i) {

    for (ii = i - 1 ; ii >= 0 ; --ii) {
      double x = Aat(ii,i) ;

      for (j = numRows ; j < (signed)numColumns ; ++j) {
        Aat(ii,j) -= x * Aat(i,j) ;
      }
    }
  }

#if 0
  {
    VL_PRINTF("after substitution\n");

    vl_index iii, jjj ;
    for (iii = 0 ; iii < 2 ; ++iii) {
      for (jjj = 0 ; jjj < 3 ; ++jjj) {
        VL_PRINTF("%5.2g ", Aat(iii,jjj)) ;

      }
      VL_PRINTF("\n") ;
    }
    VL_PRINTF("\n") ;
  }
#endif


  return VL_ERR_OK ;
}


#endif

#undef VL_MATHOP_INSTANTIATING
