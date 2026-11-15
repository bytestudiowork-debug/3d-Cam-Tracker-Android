




/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/






















#ifndef VL_IMOPV_INSTANTIATING

#include "imopv.h"
#include "imopv_sse2.h"
#include "mathop.h"

#define FLT VL_TYPE_FLOAT
#define VL_IMOPV_INSTANTIATING
#include "imopv.c"

#define FLT VL_TYPE_DOUBLE
#define VL_IMOPV_INSTANTIATING
#include "imopv.c"

#define FLT VL_TYPE_UINT32
#define VL_IMOPV_INSTANTIATING
#include "imopv.c"

#define FLT VL_TYPE_INT32
#define VL_IMOPV_INSTANTIATING
#include "imopv.c"


#endif

#if defined(VL_IMOPV_INSTANTIATING) || defined(__DOXYGEN__)

#include "float.h"





#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)


















































VL_EXPORT void
VL_XCAT(vl_imconvcol_v, SFX)
(T* dst, vl_size dst_stride,
 T const* src,
 vl_size src_width, vl_size src_height, vl_size src_stride,
 T const* filt, vl_index filt_begin, vl_index filt_end,
 int step, unsigned int flags)
{
  vl_index x = 0 ;
  vl_index y ;
  vl_index dheight = (src_height - 1) / step + 1 ;
  vl_bool transp = flags & VL_TRANSPOSE ;
  vl_bool zeropad = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;


#ifndef VL_DISABLE_SSE2
  if (vl_cpu_has_sse2() && vl_get_simd_enabled()) {
    VL_XCAT3(_vl_imconvcol_v,SFX,_sse2)
    (dst,dst_stride,
     src,src_width,src_height,src_stride,
     filt,filt_begin,filt_end,
     step,flags) ;
    return ;
  }
#endif


  filt += filt_end - filt_begin ;

  while (x < (signed)src_width) {









    T const *filti ;
    vl_index stop ;

    for (y = 0 ; y < (signed)src_height ; y += step) {
      T acc = 0 ;
      T v = 0, c ;
      T const* srci ;

      filti = filt ;
      stop = filt_end - y ;
      srci = src + x - stop * src_stride ;

      if (stop > 0) {
        if (zeropad) {
          v = 0 ;
        } else {
          v = *(src + x) ;
        }
        while (filti > filt - stop) {
          c = *filti-- ;
          acc += v * c ;
          srci += src_stride ;
        }
      }

      stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
      while (filti > filt - stop) {
        v = *srci ;
        c = *filti-- ;
        acc += v * c ;
        srci += src_stride ;
      }

      if (zeropad) v = 0 ;

      stop = filt_end - filt_begin + 1 ;
      while (filti > filt - stop) {
        c = *filti-- ;
        acc += v * c ;
      }

      if (transp) {
        *dst = acc ; dst += 1 ;
      } else {
        *dst = acc ; dst += dst_stride ;
      }
    }
    if (transp) {
      dst += 1 * dst_stride - dheight * 1 ;
    } else {
      dst += 1 * 1 - dheight * dst_stride ;
    }
    x += 1 ;
  }
}


#endif





#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)










































































































VL_EXPORT void
VL_XCAT(vl_image_distance_transform_,SFX)
(T const * image,
 vl_size numColumns,
 vl_size numRows,
 vl_size columnStride,
 vl_size rowStride,
 T * distanceTransform,
 vl_uindex * indexes,
 T coeff,
 T offset)
{








  vl_uindex x, y ;
  T * from = vl_malloc (sizeof(T) * (numColumns + 1)) ;
  T * base = vl_malloc (sizeof(T) * numColumns) ;
  vl_uindex * baseIndexes = vl_malloc (sizeof(vl_uindex) * numColumns) ;
  vl_uindex * which = vl_malloc (sizeof(vl_uindex) * numColumns) ;
  vl_uindex num = 0 ;

  for (y = 0 ; y < numRows ; ++y) {
    num = 0 ;
    for (x = 0 ; x < numColumns ; ++x) {
      T r = image[x  * columnStride + y * rowStride] ;
      T x2 = x * x ;
#if (FLT == VL_TYPE_FLOAT)
      T from_ = - VL_INFINITY_F ;
#else
      T from_ = - VL_INFINITY_D ;
#endif











      while (num >= 1) {
        vl_uindex x_ = which[num - 1] ;
        T x2_ = x_ * x_ ;
        T r_ = image[x_ * columnStride + y * rowStride] ;
        T inters ;
        if (r == r_) {

          inters = (x + x_) / 2.0 + offset ;
        }
#if (FLT == VL_TYPE_FLOAT)
        else if (coeff > VL_EPSILON_F)
#else
        else if (coeff > VL_EPSILON_D)
#endif
        {
          inters = ((r - r_) + coeff * (x2 - x2_)) / (x - x_) / (2*coeff) + offset ;
        } else {



#if (FLT == VL_TYPE_FLOAT)
          inters = (r < r_) ? - VL_INFINITY_F : VL_INFINITY_F ;
#else
          inters = (r < r_) ? - VL_INFINITY_D : VL_INFINITY_D ;
#endif
        }
        if (inters <= from [num - 1]) {

          -- num ;
        } else {

          from_ = inters ;
          break ;
        }
      }


      which[num] = x ;
      from[num] = from_ ;
      base[num] = r ;
      if (indexes) baseIndexes[num] = indexes[x  * columnStride + y * rowStride] ;
      num ++ ;
    }

#if (FLT == VL_TYPE_FLOAT)
    from[num] = VL_INFINITY_F ;
#else
    from[num] = VL_INFINITY_D ;
#endif


    num = 0 ;
    for (x = 0 ; x < numColumns ; ++x) {
      double delta ;
      while (x >= from[num + 1]) ++ num ;
      delta = (double) x - (double) which[num] - offset ;
      distanceTransform[x  * columnStride + y * rowStride]
      = base[num] + coeff * delta * delta ;
      if (indexes) {
        indexes[x  * columnStride + y * rowStride]
        = baseIndexes[num] ;
      }
    }
  }

  vl_free (from) ;
  vl_free (which) ;
  vl_free (base) ;
  vl_free (baseIndexes) ;
}


#endif





#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)























































VL_EXPORT void
VL_XCAT(vl_imconvcoltri_, SFX)
(T * dest, vl_size destStride,
 T const * image,
 vl_size imageWidth, vl_size imageHeight, vl_size imageStride,
 vl_size filterSize,
 vl_size step, unsigned int flags)
{
  vl_index x, y, dheight ;
  vl_bool transp = flags & VL_TRANSPOSE ;
  vl_bool zeropad = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;
  T scale = (T) (1.0 / ((double)filterSize * (double)filterSize)) ;
  T * buffer = vl_malloc (sizeof(T) * (imageHeight + filterSize)) ;
  buffer += filterSize ;

  if (imageHeight == 0) {
    return  ;
  }

  x = 0 ;
  dheight = (imageHeight - 1) / step + 1 ;

  while (x < (signed)imageWidth) {
    T const * imagei ;
    imagei = image + x + imageStride * (imageHeight - 1) ;









    buffer[imageHeight - 1] = *imagei ;
    for (y = (signed)imageHeight - 2 ; y >=  0 ; --y) {
      imagei -= imageStride ;
      buffer[y] = buffer[y + 1] + *imagei ;
    }
    if (zeropad) {
      for ( ; y >= - (signed)filterSize ; --y) {
        buffer[y] = buffer[y + 1] ;
      }
    } else {
      for ( ; y >= - (signed)filterSize ; --y) {
        buffer[y] = buffer[y + 1] + *imagei ;
      }
    }


    for (y = - (signed)filterSize ;
         y < (signed)imageHeight - (signed)filterSize ; ++y) {
      buffer[y] = buffer[y] - buffer[y + filterSize] ;
    }
    if (! zeropad) {
      for (y = (signed)imageHeight - (signed)filterSize ;
           y < (signed)imageHeight ;
           ++y) {
        buffer[y] = buffer[y] - buffer[imageHeight - 1]  *
        ((signed)imageHeight - (signed)filterSize - y) ;
      }
    }


    for (y = - (signed)filterSize + 1 ;
         y < (signed)imageHeight ; ++y) {
      buffer[y] += buffer[y - 1] ;
    }


    {
      vl_size stride = transp ? 1 : destStride ;
      dest += dheight * stride ;
      for (y = step * (dheight - 1) ; y >= 0 ; y -= step) {
        dest -= stride ;
        *dest = scale * (buffer[y] - buffer[y - (signed)filterSize]) ;
      }
      dest += transp ? destStride : 1 ;
    }
    x += 1 ;
  }
  vl_free (buffer - filterSize) ;
}


#endif





#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)


















static T*
VL_XCAT(_vl_new_gaussian_fitler_,SFX)(vl_size *size, double sigma)
{
  T* filter ;
  T mass = (T)1.0 ;
  vl_index i ;
  vl_size width = vl_ceil_d(sigma * 3.0) ;
  *size = 2 * width + 1 ;

  assert(size) ;

  filter = vl_malloc((*size) * sizeof(T)) ;
  filter[width] = 1.0 ;
  for (i = 1 ; i <= (signed)width ; ++i) {
    double x = (double)i / sigma ;
    double g = exp(-0.5 * x * x) ;
    mass += g + g ;
    filter[width-i] = g ;
    filter[width+i] = g ;
  }
  for (i = 0 ; i < (signed)(*size) ; ++i) {filter[i] /= mass ;}
  return filter ;
}

VL_EXPORT void
VL_XCAT(vl_imsmooth_, SFX)
(T * smoothed, vl_size smoothedStride,
 T const *image, vl_size width, vl_size height, vl_size stride,
 double sigmax, double sigmay)
{
  T *filterx, *filtery, *buffer ;
  vl_size sizex, sizey ;

  filterx = VL_XCAT(_vl_new_gaussian_fitler_,SFX)(&sizex,sigmax) ;
  if (sigmax == sigmay) {
    filtery = filterx ;
    sizey = sizex ;
  } else {
    filtery = VL_XCAT(_vl_new_gaussian_fitler_,SFX)(&sizey,sigmay) ;
  }
  buffer = vl_malloc(width*height*sizeof(T)) ;

  VL_XCAT(vl_imconvcol_v,SFX) (buffer, height,
                               image, width, height, stride,
                               filtery,
                               -((signed)sizey-1)/2, ((signed)sizey-1)/2,
                               1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;

  VL_XCAT(vl_imconvcol_v,SFX) (smoothed, smoothedStride,
                               buffer, height, width, height,
                               filterx,
                               -((signed)sizex-1)/2, ((signed)sizex-1)/2,
                               1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;

  vl_free(buffer) ;
  vl_free(filterx) ;
  if (sigmax != sigmay) {
    vl_free(filtery) ;
  }
}


#endif





#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)

































VL_EXPORT void
VL_XCAT(vl_imgradient_, SFX)
(T * xGradient, T * yGradient,
 vl_size gradWidthStride, vl_size gradHeightStride,
 T const * image,
 vl_size imageWidth, vl_size imageHeight,
 vl_size imageStride)
{

  vl_index const xo = 1 ;
  vl_index const yo = imageStride ;
  vl_size const w = imageWidth;
  vl_size const h = imageHeight;

  T const *src, *end ;
  T *pgrad_x, *pgrad_y;
  vl_size y;

  src  = image ;
  pgrad_x = xGradient ;
  pgrad_y = yGradient ;


  *pgrad_x = src[+xo] - src[0] ;
  pgrad_x += gradWidthStride;
  *pgrad_y = src[+yo] - src[0] ;
  pgrad_y += gradWidthStride;
  src++;


  end = (src - 1) + w - 1 ;
  while (src < end) {
    *pgrad_x = 0.5 * (src[+xo] - src[-xo]) ;
    pgrad_x += gradWidthStride;
    *pgrad_y =        src[+yo] - src[0] ;
    pgrad_y += gradWidthStride;
    src++;
  }


  *pgrad_x = src[0]   - src[-xo] ;
  pgrad_x += gradWidthStride;
  *pgrad_y = src[+yo] - src[0] ;
  pgrad_y += gradWidthStride;
  src++;

  xGradient += gradHeightStride;
  pgrad_x = xGradient;
  yGradient += gradHeightStride;
  pgrad_y = yGradient;
  image += yo;
  src = image;

  for (y = 1 ; y < h -1 ; ++y) {


    *pgrad_x =        src[+xo] - src[0] ;
    pgrad_x += gradWidthStride;
    *pgrad_y = 0.5 * (src[+yo] - src[-yo]) ;
    pgrad_y += gradWidthStride;
    src++;


    end = (src - 1) + w - 1 ;
    while (src < end) {
      *pgrad_x = 0.5 * (src[+xo] - src[-xo]) ;
      pgrad_x += gradWidthStride;
      *pgrad_y = 0.5 * (src[+yo] - src[-yo]) ;
      pgrad_y += gradWidthStride;
      src++;
    }


    *pgrad_x =        src[0]   - src[-xo] ;
    pgrad_x += gradWidthStride;
    *pgrad_y = 0.5 * (src[+yo] - src[-yo]) ;
    pgrad_y += gradWidthStride;
    src++;

    xGradient += gradHeightStride;
    pgrad_x = xGradient;
    yGradient += gradHeightStride;
    pgrad_y = yGradient;
    image += yo;
    src = image;
  }


  *pgrad_x = src[+xo] - src[0] ;
  pgrad_x += gradWidthStride;
  *pgrad_y = src[  0] - src[-yo] ;
  pgrad_y += gradWidthStride;
  src++;


  end = (src - 1) + w - 1 ;
  while (src < end) {
    *pgrad_x = 0.5 * (src[+xo] - src[-xo]) ;
    pgrad_x += gradWidthStride;
    *pgrad_y =        src[0]   - src[-yo] ;
    pgrad_y += gradWidthStride;
    src++;
  }


  *pgrad_x = src[0]   - src[-xo] ;
  *pgrad_y = src[0]   - src[-yo] ;
}

#endif








































#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)

VL_EXPORT void
VL_XCAT(vl_imgradient_polar_, SFX)
(T * gradientModulus, T * gradientAngle,
 vl_size gradientHorizontalStride, vl_size gradHeightStride,
 T const* image,
 vl_size imageWidth, vl_size imageHeight, vl_size imageStride)
{

  vl_index const xo = 1 ;
  vl_index const yo = imageStride ;
  vl_size const w = imageWidth;
  vl_size const h = imageHeight;

  T const *src, *end;
  T *pgrad_angl, *pgrad_ampl;
  T gx, gy ;
  vl_size y;

#define SAVE_BACK                                                    \
*pgrad_ampl = vl_fast_sqrt_f (gx*gx + gy*gy) ;                       \
pgrad_ampl += gradientHorizontalStride ;                             \
*pgrad_angl = vl_mod_2pi_f   (vl_fast_atan2_f (gy, gx) + 2*VL_PI) ;  \
pgrad_angl += gradientHorizontalStride ;                             \
++src ;                                                              \

  src  = image ;
  pgrad_angl = gradientAngle ;
  pgrad_ampl = gradientModulus ;


  gx = src[+xo] - src[0] ;
  gy = src[+yo] - src[0] ;
  SAVE_BACK ;


  end = (src - 1) + w - 1 ;
  while (src < end) {
    gx = 0.5 * (src[+xo] - src[-xo]) ;
    gy =        src[+yo] - src[0] ;
    SAVE_BACK ;
  }


  gx = src[0]   - src[-xo] ;
  gy = src[+yo] - src[0] ;
  SAVE_BACK ;

  gradientModulus += gradHeightStride;
  pgrad_ampl = gradientModulus;
  gradientAngle += gradHeightStride;
  pgrad_angl = gradientAngle;
  image += imageStride;
  src = image;

  for (y = 1 ; y < h -1 ; ++y) {


    gx =        src[+xo] - src[0] ;
    gy = 0.5 * (src[+yo] - src[-yo]) ;
    SAVE_BACK ;


    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;
    }


    gx =        src[0]   - src[-xo] ;
    gy = 0.5 * (src[+yo] - src[-yo]) ;
    SAVE_BACK ;

    gradientModulus += gradHeightStride;
    pgrad_ampl = gradientModulus;
    gradientAngle += gradHeightStride;
    pgrad_angl = gradientAngle;
    image += imageStride;
    src = image;
  }


  gx = src[+xo] - src[0] ;
  gy = src[  0] - src[-yo] ;
  SAVE_BACK ;


  end = (src - 1) + w - 1 ;
  while (src < end) {
    gx = 0.5 * (src[+xo] - src[-xo]) ;
    gy =        src[0]   - src[-yo] ;
    SAVE_BACK ;
  }


  gx = src[0]   - src[-xo] ;
  gy = src[0]   - src[-yo] ;
  SAVE_BACK ;

}

#endif
































































VL_EXPORT void
VL_XCAT(vl_imintegral_, SFX)
(T * integral, vl_size integralStride,
 T const * image,
 vl_size imageWidth, vl_size imageHeight, vl_size imageStride)
{
  vl_uindex x, y ;
  T temp  = 0 ;

  if (imageHeight > 0) {
    for (x = 0 ; x < imageWidth ; ++ x) {
      temp += *image++ ;
      *integral++ = temp ;
    }
  }

  for (y = 1 ; y < imageHeight ; ++ y) {
    T * integralPrev ;
    integral += integralStride - imageWidth ;
    image += imageStride - imageWidth ;
    integralPrev = integral - integralStride ;

    temp = 0 ;
    for (x = 0 ; x < imageWidth ; ++ x) {
      temp += *image++ ;
      *integral++ = *integralPrev++ + temp ;
    }
  }
}


#undef FLT
#undef VL_IMOPV_INSTANTIATING
#endif
