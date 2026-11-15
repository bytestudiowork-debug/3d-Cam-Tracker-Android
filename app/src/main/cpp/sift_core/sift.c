




/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/





































































































































































































































































































































































































































































































































































































































































#include "sift.h"
#include "imopv.h"
#include "mathop.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>


#define VL_SIFT_BILINEAR_ORIENTATIONS 1

#define EXPN_SZ  256
#define EXPN_MAX 25.0
double expn_tab [EXPN_SZ+1] ;

#define NBO 8
#define NBP 4

#define log2(x) (log(x)/VL_LOG_OF_2)












VL_INLINE double
fast_expn (double x)
{
  double a,b,r ;
  int i ;


  if (x > EXPN_MAX) return 0.0 ;

  x *= EXPN_SZ / EXPN_MAX ;
  i = (int)vl_floor_d (x) ;
  r = x - i ;
  a = expn_tab [i    ] ;
  b = expn_tab [i + 1] ;
  return a + r * (b - a) ;
}






VL_INLINE void
fast_expn_init ()
{
  int k  ;
  for(k = 0 ; k < EXPN_SZ + 1 ; ++ k) {
    expn_tab [k] = exp (- (double) k * (EXPN_MAX / EXPN_SZ)) ;
  }
}

















static void
copy_and_upsample_rows
(vl_sift_pix       *dst,
 vl_sift_pix const *src, int width, int height)
{
  int x, y ;
  vl_sift_pix a, b ;

  for(y = 0 ; y < height ; ++y) {
    b = a = *src++ ;
    for(x = 0 ; x < width - 1 ; ++x) {
      b = *src++ ;
      *dst = a ;             dst += height ;
      *dst = 0.5 * (a + b) ; dst += height ;
      a = b ;
    }
    *dst = b ; dst += height ;
    *dst = b ; dst += height ;
    dst += 1 - width * 2 * height ;
  }
}













static void
_vl_sift_smooth (VlSiftFilt * self,
                 vl_sift_pix * outputImage,
                 vl_sift_pix * tempImage,
                 vl_sift_pix const * inputImage,
                 vl_size width,
                 vl_size height,
                 double sigma)
{

  if (self->gaussFilterSigma != sigma) {
    vl_uindex j ;
    vl_sift_pix acc = 0 ;
    if (self->gaussFilter) vl_free (self->gaussFilter) ;
    self->gaussFilterWidth = VL_MAX(ceil(4.0 * sigma), 1) ;
    self->gaussFilterSigma = sigma ;
    self->gaussFilter = vl_malloc (sizeof(vl_sift_pix) * (2 * self->gaussFilterWidth + 1)) ;

    for (j = 0 ; j < 2 * self->gaussFilterWidth + 1 ; ++j) {
      vl_sift_pix d = ((vl_sift_pix)((signed)j - (signed)self->gaussFilterWidth)) / ((vl_sift_pix)sigma) ;
      self->gaussFilter[j] = (vl_sift_pix) exp (- 0.5 * (d*d)) ;
      acc += self->gaussFilter[j] ;
    }
    for (j = 0 ; j < 2 * self->gaussFilterWidth + 1 ; ++j) {
      self->gaussFilter[j] /= acc ;
    }
  }

  if (self->gaussFilterWidth == 0) {
    memcpy (outputImage, inputImage, sizeof(vl_sift_pix) * width * height) ;
    return ;
  }

  vl_imconvcol_vf (tempImage, height,
                   inputImage, width, height, width,
                   self->gaussFilter,
                   - self->gaussFilterWidth, self->gaussFilterWidth,
                   1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;

  vl_imconvcol_vf (outputImage, width,
                   tempImage, height, width, height,
                   self->gaussFilter,
                   - self->gaussFilterWidth, self->gaussFilterWidth,
                   1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;
}


















static void
copy_and_downsample
(vl_sift_pix       *dst,
 vl_sift_pix const *src,
 int width, int height, int d)
{
  int x, y ;

  d = 1 << d ;
  for(y = 0 ; y < height ; y+=d) {
    vl_sift_pix const * srcrowp = src + y * width ;
    for(x = 0 ; x < width - (d-1) ; x+=d) {
      *dst++ = *srcrowp ;
      srcrowp += d ;
    }
  }
}




















VL_EXPORT
VlSiftFilt *
vl_sift_new (int width, int height,
             int noctaves, int nlevels,
             int o_min)
{
  VlSiftFilt *f = vl_malloc (sizeof(VlSiftFilt)) ;

  int w   = VL_SHIFT_LEFT (width,  -o_min) ;
  int h   = VL_SHIFT_LEFT (height, -o_min) ;
  int nel = w * h ;


  if (noctaves < 0) {
    noctaves = VL_MAX (floor (log2 (VL_MIN(width, height))) - o_min - 3, 1) ;
  }

  f-> width   = width ;
  f-> height  = height ;
  f-> O       = noctaves ;
  f-> S       = nlevels ;
  f-> o_min   = o_min ;
  f-> s_min   = -1 ;
  f-> s_max   = nlevels + 1 ;
  f-> o_cur   = o_min ;

  f-> temp    = vl_malloc (sizeof(vl_sift_pix) * nel    ) ;
  f-> octave  = vl_malloc (sizeof(vl_sift_pix) * nel
                        * (f->s_max - f->s_min + 1)  ) ;
  f-> dog     = vl_malloc (sizeof(vl_sift_pix) * nel
                        * (f->s_max - f->s_min    )  ) ;
  f-> grad    = vl_malloc (sizeof(vl_sift_pix) * nel * 2
                        * (f->s_max - f->s_min    )  ) ;

  f-> sigman  = 0.5 ;
  f-> sigmak  = pow (2.0, 1.0 / nlevels) ;
  f-> sigma0  = 1.6 * f->sigmak ;
  f-> dsigma0 = f->sigma0 * sqrt (1.0 - 1.0 / (f->sigmak*f->sigmak)) ;

  f-> gaussFilter = NULL ;
  f-> gaussFilterSigma = 0 ;
  f-> gaussFilterWidth = 0 ;

  f-> octave_width  = 0 ;
  f-> octave_height = 0 ;

  f-> keys     = 0 ;
  f-> nkeys    = 0 ;
  f-> keys_res = 0 ;

  f-> peak_thresh = 0.0 ;
  f-> edge_thresh = 10.0 ;
  f-> norm_thresh = 0.0 ;
  f-> magnif      = 3.0 ;
  f-> windowSize  = NBP / 2 ;

  f-> grad_o  = o_min - 1 ;


  fast_expn_init () ;

  return f ;
}









VL_EXPORT
void
vl_sift_delete (VlSiftFilt* f)
{
  if (f) {
    if (f->keys) vl_free (f->keys) ;
    if (f->grad) vl_free (f->grad) ;
    if (f->dog) vl_free (f->dog) ;
    if (f->octave) vl_free (f->octave) ;
    if (f->temp) vl_free (f->temp) ;
    if (f->gaussFilter) vl_free (f->gaussFilter) ;
    vl_free (f) ;
  }
}

















VL_EXPORT
int
vl_sift_process_first_octave (VlSiftFilt *f, vl_sift_pix const *im)
{
  int o, s, h, w ;
  double sa, sb ;
  vl_sift_pix *octave ;


  vl_sift_pix *temp   = f-> temp ;
  int width           = f-> width ;
  int height          = f-> height ;
  int o_min           = f-> o_min ;
  int s_min           = f-> s_min ;
  int s_max           = f-> s_max ;
  double sigma0       = f-> sigma0 ;
  double sigmak       = f-> sigmak ;
  double sigman       = f-> sigman ;
  double dsigma0      = f-> dsigma0 ;


  f->o_cur = o_min ;
  f->nkeys = 0 ;
  w = f-> octave_width  = VL_SHIFT_LEFT(f->width,  - f->o_cur) ;
  h = f-> octave_height = VL_SHIFT_LEFT(f->height, - f->o_cur) ;


  if (f->O == 0)
    return VL_ERR_EOF ;











  octave = vl_sift_get_octave (f, s_min) ;

  if (o_min < 0) {

    copy_and_upsample_rows (temp,   im,   width,      height) ;
    copy_and_upsample_rows (octave, temp, height, 2 * width ) ;


    for(o = -1 ; o > o_min ; --o) {
      copy_and_upsample_rows (temp, octave,
                              width << -o,      height << -o ) ;
      copy_and_upsample_rows (octave, temp,
                              width << -o, 2 * (height << -o)) ;
    }
  }
  else if (o_min > 0) {

    copy_and_downsample (octave, im, width, height, o_min) ;
  }
  else {

    memcpy(octave, im, sizeof(vl_sift_pix) * width * height) ;
  }







  sa = sigma0 * pow (sigmak,   s_min) ;
  sb = sigman * pow (2.0,    - o_min) ;

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb) ;
    _vl_sift_smooth (f, octave, temp, octave, w, h, sd) ;
  }





  for(s = s_min + 1 ; s <= s_max ; ++s) {
    double sd = dsigma0 * pow (sigmak, s) ;
    _vl_sift_smooth (f, vl_sift_get_octave(f, s), temp,
                     vl_sift_get_octave(f, s - 1), w, h, sd) ;
  }

  return VL_ERR_OK ;
}
















VL_EXPORT
int
vl_sift_process_next_octave (VlSiftFilt *f)
{

  int s, h, w, s_best ;
  double sa, sb ;
  vl_sift_pix *octave, *pt ;


  vl_sift_pix *temp   = f-> temp ;
  int O               = f-> O ;
  int S               = f-> S ;
  int o_min           = f-> o_min ;
  int s_min           = f-> s_min ;
  int s_max           = f-> s_max ;
  double sigma0       = f-> sigma0 ;
  double sigmak       = f-> sigmak ;
  double dsigma0      = f-> dsigma0 ;


  if (f->o_cur == o_min + O - 1)
    return VL_ERR_EOF ;


  s_best = VL_MIN(s_min + S, s_max) ;
  w      = vl_sift_get_octave_width  (f) ;
  h      = vl_sift_get_octave_height (f) ;
  pt     = vl_sift_get_octave        (f, s_best) ;
  octave = vl_sift_get_octave        (f, s_min) ;


  copy_and_downsample (octave, pt, w, h, 1) ;

  f-> o_cur            += 1 ;
  f-> nkeys             = 0 ;
  w = f-> octave_width  = VL_SHIFT_LEFT(f->width,  - f->o_cur) ;
  h = f-> octave_height = VL_SHIFT_LEFT(f->height, - f->o_cur) ;

  sa = sigma0 * powf (sigmak, s_min     ) ;
  sb = sigma0 * powf (sigmak, s_best - S) ;

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb) ;
    _vl_sift_smooth (f, octave, temp, octave, w, h, sd) ;
  }





  for(s = s_min + 1 ; s <= s_max ; ++s) {
    double sd = dsigma0 * pow (sigmak, s) ;
    _vl_sift_smooth (f, vl_sift_get_octave(f, s), temp,
                     vl_sift_get_octave(f, s - 1), w, h, sd) ;
  }

  return VL_ERR_OK ;
}











VL_EXPORT
void
vl_sift_detect (VlSiftFilt * f)
{
  vl_sift_pix* dog   = f-> dog ;
  int          s_min = f-> s_min ;
  int          s_max = f-> s_max ;
  int          w     = f-> octave_width ;
  int          h     = f-> octave_height ;
  double       te    = f-> edge_thresh ;
  double       tp    = f-> peak_thresh ;

  int const    xo    = 1 ;
  int const    yo    = w ;
  int const    so    = w * h ;

  double       xper  = pow (2.0, f->o_cur) ;

  int x, y, s, i, ii, jj ;
  vl_sift_pix *pt, v ;
  VlSiftKeypoint *k ;


  f-> nkeys = 0 ;


  pt = f-> dog ;
  for (s = s_min ; s <= s_max - 1 ; ++s) {
    vl_sift_pix* src_a = vl_sift_get_octave (f, s    ) ;
    vl_sift_pix* src_b = vl_sift_get_octave (f, s + 1) ;
    vl_sift_pix* end_a = src_a + w * h ;
    while (src_a != end_a) {
      *pt++ = *src_b++ - *src_a++ ;
    }
  }






  pt  = dog + xo + yo + so ;

  for(s = s_min + 1 ; s <= s_max - 2 ; ++s) {
    for(y = 1 ; y < h - 1 ; ++y) {
      for(x = 1 ; x < w - 1 ; ++x) {
        v = *pt ;

#define CHECK_NEIGHBORS(CMP,SGN)                    \
        ( v CMP ## = SGN 0.8 * tp &&                \
          v CMP *(pt + xo) &&                       \
          v CMP *(pt - xo) &&                       \
          v CMP *(pt + so) &&                       \
          v CMP *(pt - so) &&                       \
          v CMP *(pt + yo) &&                       \
          v CMP *(pt - yo) &&                       \
                                                    \
          v CMP *(pt + yo + xo) &&                  \
          v CMP *(pt + yo - xo) &&                  \
          v CMP *(pt - yo + xo) &&                  \
          v CMP *(pt - yo - xo) &&                  \
                                                    \
          v CMP *(pt + xo      + so) &&             \
          v CMP *(pt - xo      + so) &&             \
          v CMP *(pt + yo      + so) &&             \
          v CMP *(pt - yo      + so) &&             \
          v CMP *(pt + yo + xo + so) &&             \
          v CMP *(pt + yo - xo + so) &&             \
          v CMP *(pt - yo + xo + so) &&             \
          v CMP *(pt - yo - xo + so) &&             \
                                                    \
          v CMP *(pt + xo      - so) &&             \
          v CMP *(pt - xo      - so) &&             \
          v CMP *(pt + yo      - so) &&             \
          v CMP *(pt - yo      - so) &&             \
          v CMP *(pt + yo + xo - so) &&             \
          v CMP *(pt + yo - xo - so) &&             \
          v CMP *(pt - yo + xo - so) &&             \
          v CMP *(pt - yo - xo - so) )

        if (CHECK_NEIGHBORS(>,+) ||
            CHECK_NEIGHBORS(<,-) ) {


          if (f->nkeys >= f->keys_res) {
            f->keys_res += 500 ;
            if (f->keys) {
              f->keys = vl_realloc (f->keys,
                                    f->keys_res *
                                    sizeof(VlSiftKeypoint)) ;
            } else {
              f->keys = vl_malloc (f->keys_res *
                                   sizeof(VlSiftKeypoint)) ;
            }
          }

          k = f->keys + (f->nkeys ++) ;

          k-> ix = x ;
          k-> iy = y ;
          k-> is = s ;
        }
        pt += 1 ;
      }
      pt += 2 ;
    }
    pt += 2 * yo ;
  }






  k = f->keys ;

  for (i = 0 ; i < f->nkeys ; ++i) {

    int x = f-> keys [i] .ix ;
    int y = f-> keys [i] .iy ;
    int s = f-> keys [i]. is ;

    double Dx=0,Dy=0,Ds=0,Dxx=0,Dyy=0,Dss=0,Dxy=0,Dxs=0,Dys=0 ;
    double A [3*3], b [3] ;

    int dx = 0 ;
    int dy = 0 ;

    int iter, i, j ;

    for (iter = 0 ; iter < 5 ; ++iter) {

      x += dx ;
      y += dy ;

      pt = dog
        + xo * x
        + yo * y
        + so * (s - s_min) ;


#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))


#define Aat(i,j)     (A[(i)+(j)*3])


      Dx = 0.5 * (at(+1,0,0) - at(-1,0,0)) ;
      Dy = 0.5 * (at(0,+1,0) - at(0,-1,0));
      Ds = 0.5 * (at(0,0,+1) - at(0,0,-1)) ;


      Dxx = (at(+1,0,0) + at(-1,0,0) - 2.0 * at(0,0,0)) ;
      Dyy = (at(0,+1,0) + at(0,-1,0) - 2.0 * at(0,0,0)) ;
      Dss = (at(0,0,+1) + at(0,0,-1) - 2.0 * at(0,0,0)) ;

      Dxy = 0.25 * ( at(+1,+1,0) + at(-1,-1,0) - at(-1,+1,0) - at(+1,-1,0) ) ;
      Dxs = 0.25 * ( at(+1,0,+1) + at(-1,0,-1) - at(-1,0,+1) - at(+1,0,-1) ) ;
      Dys = 0.25 * ( at(0,+1,+1) + at(0,-1,-1) - at(0,-1,+1) - at(0,+1,-1) ) ;


      Aat(0,0) = Dxx ;
      Aat(1,1) = Dyy ;
      Aat(2,2) = Dss ;
      Aat(0,1) = Aat(1,0) = Dxy ;
      Aat(0,2) = Aat(2,0) = Dxs ;
      Aat(1,2) = Aat(2,1) = Dys ;

      b[0] = - Dx ;
      b[1] = - Dy ;
      b[2] = - Ds ;


      for(j = 0 ; j < 3 ; ++j) {
        double maxa    = 0 ;
        double maxabsa = 0 ;
        int    maxi    = -1 ;
        double tmp ;


        for (i = j ; i < 3 ; ++i) {
          double a    = Aat (i,j) ;
          double absa = vl_abs_d (a) ;
          if (absa > maxabsa) {
            maxa    = a ;
            maxabsa = absa ;
            maxi    = i ;
          }
        }


        if (maxabsa < 1e-10f) {
          b[0] = 0 ;
          b[1] = 0 ;
          b[2] = 0 ;
          break ;
        }

        i = maxi ;


        for(jj = j ; jj < 3 ; ++jj) {
          tmp = Aat(i,jj) ; Aat(i,jj) = Aat(j,jj) ; Aat(j,jj) = tmp ;
          Aat(j,jj) /= maxa ;
        }
        tmp = b[j] ; b[j] = b[i] ; b[i] = tmp ;
        b[j] /= maxa ;


        for (ii = j+1 ; ii < 3 ; ++ii) {
          double x = Aat(ii,j) ;
          for (jj = j ; jj < 3 ; ++jj) {
            Aat(ii,jj) -= x * Aat(j,jj) ;
          }
          b[ii] -= x * b[j] ;
        }
      }


      for (i = 2 ; i > 0 ; --i) {
        double x = b[i] ;
        for (ii = i-1 ; ii >= 0 ; --ii) {
          b[ii] -= x * Aat(ii,i) ;
        }
      }






      dx= ((b[0] >  0.6 && x < w - 2) ?  1 : 0)
        + ((b[0] < -0.6 && x > 1    ) ? -1 : 0) ;

      dy= ((b[1] >  0.6 && y < h - 2) ?  1 : 0)
        + ((b[1] < -0.6 && y > 1    ) ? -1 : 0) ;

      if (dx == 0 && dy == 0) break ;
    }


    {
      double val   = at(0,0,0)
        + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]) ;
      double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy) ;
      double xn = x + b[0] ;
      double yn = y + b[1] ;
      double sn = s + b[2] ;

      vl_bool good =
        vl_abs_d (val)  > tp                  &&
        score           < (te+1)*(te+1)/te    &&
        score           >= 0                  &&
        vl_abs_d (b[0]) <  1.5                &&
        vl_abs_d (b[1]) <  1.5                &&
        vl_abs_d (b[2]) <  1.5                &&
        xn              >= 0                  &&
        xn              <= w - 1              &&
        yn              >= 0                  &&
        yn              <= h - 1              &&
        sn              >= s_min              &&
        sn              <= s_max ;

      if (good) {
        k-> o     = f->o_cur ;
        k-> ix    = x ;
        k-> iy    = y ;
        k-> is    = s ;
        k-> s     = sn ;
        k-> x     = xn * xper ;
        k-> y     = yn * xper ;
        k-> sigma = f->sigma0 * pow (2.0, sn/f->S) * xper ;
        ++ k ;
      }

    }
  }


  f-> nkeys = (int)(k - f->keys) ;
}














static void
update_gradient (VlSiftFilt *f)
{
  int       s_min = f->s_min ;
  int       s_max = f->s_max ;
  int       w     = vl_sift_get_octave_width  (f) ;
  int       h     = vl_sift_get_octave_height (f) ;
  int const xo    = 1 ;
  int const yo    = w ;
  int const so    = h * w ;
  int y, s ;

  if (f->grad_o == f->o_cur) return ;

  for (s  = s_min + 1 ;
       s <= s_max - 2 ; ++ s) {

    vl_sift_pix *src, *end, *grad, gx, gy ;

#define SAVE_BACK                                                       \
    *grad++ = vl_fast_sqrt_f (gx*gx + gy*gy) ;                          \
    *grad++ = vl_mod_2pi_f   (vl_fast_atan2_f (gy, gx) + 2*VL_PI) ;     \
    ++src ;                                                             \

    src  = vl_sift_get_octave (f,s) ;
    grad = f->grad + 2 * so * (s - s_min -1) ;


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
  f->grad_o = f->o_cur ;
}

























VL_EXPORT
int
vl_sift_calc_keypoint_orientations (VlSiftFilt *f,
                                    double angles [4],
                                    VlSiftKeypoint const *k)
{
  double const winf   = 1.5 ;
  double       xper   = pow (2.0, f->o_cur) ;

  int          w      = f-> octave_width ;
  int          h      = f-> octave_height ;
  int const    xo     = 2 ;
  int const    yo     = 2 * w ;
  int const    so     = 2 * w * h ;
  double       x      = k-> x     / xper ;
  double       y      = k-> y     / xper ;
  double       sigma  = k-> sigma / xper ;

  int          xi     = (int) (x + 0.5) ;
  int          yi     = (int) (y + 0.5) ;
  int          si     = k-> is ;

  double const sigmaw = winf * sigma ;
  int          W      = VL_MAX(floor (3.0 * sigmaw), 1) ;

  int          nangles= 0 ;

  enum {nbins = 36} ;

  double hist [nbins], maxh ;
  vl_sift_pix const * pt ;
  int xs, ys, iter, i ;


  if(k->o != f->o_cur)
    return 0 ;


  if(xi < 0            ||
     xi > w - 1        ||
     yi < 0            ||
     yi > h - 1        ||
     si < f->s_min + 1 ||
     si > f->s_max - 2  ) {
    return 0 ;
  }


  update_gradient (f) ;


  memset (hist, 0, sizeof(double) * nbins) ;


  pt = f-> grad + xo*xi + yo*yi + so*(si - f->s_min - 1) ;

#undef  at
#define at(dx,dy) (*(pt + xo * (dx) + yo * (dy)))

  for(ys  =  VL_MAX (- W,       - yi) ;
      ys <=  VL_MIN (+ W, h - 1 - yi) ; ++ys) {

    for(xs  = VL_MAX (- W,       - xi) ;
        xs <= VL_MIN (+ W, w - 1 - xi) ; ++xs) {


      double dx = (double)(xi + xs) - x;
      double dy = (double)(yi + ys) - y;
      double r2 = dx*dx + dy*dy ;
      double wgt, mod, ang, fbin ;


      if (r2 >= W*W + 0.6) continue ;

      wgt  = fast_expn (r2 / (2*sigmaw*sigmaw)) ;
      mod  = *(pt + xs*xo + ys*yo    ) ;
      ang  = *(pt + xs*xo + ys*yo + 1) ;
      fbin = nbins * ang / (2 * VL_PI) ;

#if defined(VL_SIFT_BILINEAR_ORIENTATIONS)
      {
        int bin = (int) vl_floor_d (fbin - 0.5) ;
        double rbin = fbin - bin - 0.5 ;
        hist [(bin + nbins) % nbins] += (1 - rbin) * mod * wgt ;
        hist [(bin + 1    ) % nbins] += (    rbin) * mod * wgt ;
      }
#else
      {
        int    bin  = vl_floor_d (fbin) ;
        bin = vl_floor_d (nbins * ang / (2*VL_PI)) ;
        hist [(bin) % nbins] += mod * wgt ;
      }
#endif

    }
  }


  for (iter = 0; iter < 6; iter ++) {
    double prev  = hist [nbins - 1] ;
    double first = hist [0] ;
    int i ;
    for (i = 0; i < nbins - 1; i++) {
      double newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
      prev = hist[i] ;
      hist[i] = newh ;
    }
    hist[i] = (prev + hist[i] + first) / 3.0 ;
  }


  maxh = 0 ;
  for (i = 0 ; i < nbins ; ++i)
    maxh = VL_MAX (maxh, hist [i]) ;


  nangles = 0 ;
  for(i = 0 ; i < nbins ; ++i) {
    double h0 = hist [i] ;
    double hm = hist [(i - 1 + nbins) % nbins] ;
    double hp = hist [(i + 1 + nbins) % nbins] ;


    if (h0 > 0.8*maxh && h0 > hm && h0 > hp) {


      double di = - 0.5 * (hp - hm) / (hp + hm - 2 * h0) ;
      double th = 2 * VL_PI * (i + di + 0.5) / nbins ;
      angles [ nangles++ ] = th ;
      if( nangles == 4 )
        goto enough_angles ;
    }
  }
 enough_angles:
  return nangles ;
}









VL_INLINE vl_sift_pix
normalize_histogram
(vl_sift_pix *begin, vl_sift_pix *end)
{
  vl_sift_pix* iter ;
  vl_sift_pix  norm = 0.0 ;

  for (iter = begin ; iter != end ; ++ iter)
    norm += (*iter) * (*iter) ;

  norm = vl_fast_sqrt_f (norm) + VL_EPSILON_F ;

  for (iter = begin; iter != end ; ++ iter)
    *iter /= norm ;

  return norm;
}

































VL_EXPORT
void
vl_sift_calc_raw_descriptor (VlSiftFilt const *f,
                             vl_sift_pix const* grad,
                             vl_sift_pix *descr,
                             int width, int height,
                             double x, double y,
                             double sigma,
                             double angle0)
{
  double const magnif = f-> magnif ;

  int          w      = width ;
  int          h      = height ;
  int const    xo     = 2 ;
  int const    yo     = 2 * w ;

  int          xi     = (int) (x + 0.5) ;
  int          yi     = (int) (y + 0.5) ;

  double const st0    = sin (angle0) ;
  double const ct0    = cos (angle0) ;
  double const SBP    = magnif * sigma + VL_EPSILON_D ;
  int    const W      = floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  int const binto = 1 ;
  int const binyo = NBO * NBP ;
  int const binxo = NBO ;

  int bin, dxi, dyi ;
  vl_sift_pix const *pt ;
  vl_sift_pix       *dpt ;


  if(xi    <  0               ||
     xi    >= w               ||
     yi    <  0               ||
     yi    >= h -    1        )
    return ;


  memset (descr, 0, sizeof(vl_sift_pix) * NBO*NBP*NBP) ;




  pt  = grad + xi*xo + yi*yo ;
  dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo ;

#undef atd
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)





  for(dyi =  VL_MAX(- W,   - yi   ) ;
      dyi <= VL_MIN(+ W, h - yi -1) ; ++ dyi) {

    for(dxi =  VL_MAX(- W,   - xi   ) ;
        dxi <= VL_MIN(+ W, w - xi -1) ; ++ dxi) {


      vl_sift_pix mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      vl_sift_pix angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
      vl_sift_pix theta = vl_mod_2pi_f (angle - angle0) ;


      vl_sift_pix dx = xi + dxi - x;
      vl_sift_pix dy = yi + dyi - y;



      vl_sift_pix nx = ( ct0 * dx + st0 * dy) / SBP ;
      vl_sift_pix ny = (-st0 * dx + ct0 * dy) / SBP ;
      vl_sift_pix nt = NBO * theta / (2 * VL_PI) ;





      vl_sift_pix const wsigma = f->windowSize ;
      vl_sift_pix win = fast_expn
        ((nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;



      int         binx = (int)vl_floor_f (nx - 0.5) ;
      int         biny = (int)vl_floor_f (ny - 0.5) ;
      int         bint = (int)vl_floor_f (nt) ;
      vl_sift_pix rbinx = nx - (binx + 0.5) ;
      vl_sift_pix rbiny = ny - (biny + 0.5) ;
      vl_sift_pix rbint = nt - bint ;
      int         dbinx ;
      int         dbiny ;
      int         dbint ;


      for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
        for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
          for(dbint = 0 ; dbint < 2 ; ++dbint) {

            if (binx + dbinx >= - (NBP/2) &&
                binx + dbinx <    (NBP/2) &&
                biny + dbiny >= - (NBP/2) &&
                biny + dbiny <    (NBP/2) ) {
              vl_sift_pix weight = win
                * mod
                * vl_abs_f (1 - dbinx - rbinx)
                * vl_abs_f (1 - dbiny - rbiny)
                * vl_abs_f (1 - dbint - rbint) ;

              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            }
          }
        }
      }
    }
  }


  if(1) {


    vl_sift_pix norm = normalize_histogram (descr, descr + NBO*NBP*NBP) ;







    int numSamples =
      (VL_MIN(W, w - xi -1) - VL_MAX(-W, - xi) + 1) *
      (VL_MIN(W, h - yi -1) - VL_MAX(-W, - yi) + 1) ;

    if(f-> norm_thresh && norm < f-> norm_thresh * numSamples) {
        for(bin = 0; bin < NBO*NBP*NBP ; ++ bin)
            descr [bin] = 0;
    }
    else {

      for(bin = 0; bin < NBO*NBP*NBP ; ++ bin) {
        if (descr [bin] > 0.2) descr [bin] = 0.2;
      }


      normalize_histogram (descr, descr + NBO*NBP*NBP) ;
    }
  }
}

















VL_EXPORT
void
vl_sift_calc_keypoint_descriptor (VlSiftFilt *f,
                                  vl_sift_pix *descr,
                                  VlSiftKeypoint const* k,
                                  double angle0)
{
















  double const magnif      = f-> magnif ;

  double       xper        = pow (2.0, f->o_cur) ;

  int          w           = f-> octave_width ;
  int          h           = f-> octave_height ;
  int const    xo          = 2 ;
  int const    yo          = 2 * w ;
  int const    so          = 2 * w * h ;
  double       x           = k-> x     / xper ;
  double       y           = k-> y     / xper ;
  double       sigma       = k-> sigma / xper ;

  int          xi          = (int) (x + 0.5) ;
  int          yi          = (int) (y + 0.5) ;
  int          si          = k-> is ;

  double const st0         = sin (angle0) ;
  double const ct0         = cos (angle0) ;
  double const SBP         = magnif * sigma + VL_EPSILON_D ;
  int    const W           = floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  int const binto = 1 ;
  int const binyo = NBO * NBP ;
  int const binxo = NBO ;

  int bin, dxi, dyi ;
  vl_sift_pix const *pt ;
  vl_sift_pix       *dpt ;


  if(k->o  != f->o_cur        ||
     xi    <  0               ||
     xi    >= w               ||
     yi    <  0               ||
     yi    >= h -    1        ||
     si    <  f->s_min + 1    ||
     si    >  f->s_max - 2     )
    return ;


  update_gradient (f) ;




  memset (descr, 0, sizeof(vl_sift_pix) * NBO*NBP*NBP) ;




  pt  = f->grad + xi*xo + yi*yo + (si - f->s_min - 1)*so ;
  dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo ;

#undef atd
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)





  for(dyi =  VL_MAX (- W, 1 - yi    ) ;
      dyi <= VL_MIN (+ W, h - yi - 2) ; ++ dyi) {

    for(dxi =  VL_MAX (- W, 1 - xi    ) ;
        dxi <= VL_MIN (+ W, w - xi - 2) ; ++ dxi) {


      vl_sift_pix mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      vl_sift_pix angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
      vl_sift_pix theta = vl_mod_2pi_f (angle - angle0) ;


      vl_sift_pix dx = xi + dxi - x;
      vl_sift_pix dy = yi + dyi - y;



      vl_sift_pix nx = ( ct0 * dx + st0 * dy) / SBP ;
      vl_sift_pix ny = (-st0 * dx + ct0 * dy) / SBP ;
      vl_sift_pix nt = NBO * theta / (2 * VL_PI) ;





      vl_sift_pix const wsigma = f->windowSize ;
      vl_sift_pix win = fast_expn
        ((nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;



      int         binx = (int)vl_floor_f (nx - 0.5) ;
      int         biny = (int)vl_floor_f (ny - 0.5) ;
      int         bint = (int)vl_floor_f (nt) ;
      vl_sift_pix rbinx = nx - (binx + 0.5) ;
      vl_sift_pix rbiny = ny - (biny + 0.5) ;
      vl_sift_pix rbint = nt - bint ;
      int         dbinx ;
      int         dbiny ;
      int         dbint ;


      for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
        for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
          for(dbint = 0 ; dbint < 2 ; ++dbint) {

            if (binx + dbinx >= - (NBP/2) &&
                binx + dbinx <    (NBP/2) &&
                biny + dbiny >= - (NBP/2) &&
                biny + dbiny <    (NBP/2) ) {
              vl_sift_pix weight = win
                * mod
                * vl_abs_f (1 - dbinx - rbinx)
                * vl_abs_f (1 - dbiny - rbiny)
                * vl_abs_f (1 - dbint - rbint) ;

              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            }
          }
        }
      }
    }
  }


  if(1) {


    vl_sift_pix norm = normalize_histogram (descr, descr + NBO*NBP*NBP) ;


    if(f-> norm_thresh && norm < f-> norm_thresh) {
        for(bin = 0; bin < NBO*NBP*NBP ; ++ bin)
            descr [bin] = 0;
    }
    else {


      for(bin = 0; bin < NBO*NBP*NBP ; ++ bin) {
        if (descr [bin] > 0.2) descr [bin] = 0.2;
      }


      normalize_histogram (descr, descr + NBO*NBP*NBP) ;
    }
  }

}



































































VL_EXPORT
void
vl_sift_keypoint_init (VlSiftFilt const *f,
                       VlSiftKeypoint *k,
                       double x,
                       double y,
                       double sigma)
{
  int    o, ix, iy, is ;
  double s, phi, xper ;

  phi = log2 ((sigma + VL_EPSILON_D) / f->sigma0) ;
  o   = (int)vl_floor_d (phi -  ((double) f->s_min + 0.5) / f->S) ;
  o   = VL_MIN (o, f->o_min + f->O - 1) ;
  o   = VL_MAX (o, f->o_min           ) ;
  s   = f->S * (phi - o) ;

  is  = (int)(s + 0.5) ;
  is  = VL_MIN(is, f->s_max - 2) ;
  is  = VL_MAX(is, f->s_min + 1) ;

  xper = pow (2.0, o) ;
  ix   = (int)(x / xper + 0.5) ;
  iy   = (int)(y / xper + 0.5) ;

  k -> o  = o ;

  k -> ix = ix ;
  k -> iy = iy ;
  k -> is = is ;

  k -> x = x ;
  k -> y = y ;
  k -> s = s ;

  k->sigma = sigma ;
}
