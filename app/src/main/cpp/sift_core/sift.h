




/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_SIFT_H
#define VL_SIFT_H

#include <stdio.h>
#include "generic.h"


typedef float vl_sift_pix ;








typedef struct _VlSiftKeypoint
{
  int o ;

  int ix ;
  int iy ;
  int is ;

  float x ;
  float y ;
  float s ;
  float sigma ;
} VlSiftKeypoint ;







typedef struct _VlSiftFilt
{
  double sigman ;
  double sigma0 ;
  double sigmak ;
  double dsigma0 ;

  int width ;
  int height ;
  int O ;
  int S ;
  int o_min ;
  int s_min ;
  int s_max ;
  int o_cur ;

  vl_sift_pix *temp ;
  vl_sift_pix *octave ;
  vl_sift_pix *dog ;
  int octave_width ;
  int octave_height ;

  vl_sift_pix *gaussFilter ;
  double gaussFilterSigma ;
  vl_size gaussFilterWidth ;

  VlSiftKeypoint* keys ;
  int nkeys ;
  int keys_res ;

  double peak_thresh ;
  double edge_thresh ;
  double norm_thresh ;
  double magnif ;
  double windowSize ;

  vl_sift_pix *grad ;
  int grad_o ;

} VlSiftFilt ;




VL_EXPORT
VlSiftFilt*  vl_sift_new    (int width, int height,
                             int noctaves, int nlevels,
                             int o_min) ;
VL_EXPORT
void         vl_sift_delete (VlSiftFilt *f) ;






VL_EXPORT
int   vl_sift_process_first_octave       (VlSiftFilt *f,
                                          vl_sift_pix const *im) ;

VL_EXPORT
int   vl_sift_process_next_octave        (VlSiftFilt *f) ;

VL_EXPORT
void  vl_sift_detect                     (VlSiftFilt *f) ;

VL_EXPORT
int   vl_sift_calc_keypoint_orientations (VlSiftFilt *f,
                                          double angles [4],
                                          VlSiftKeypoint const*k);
VL_EXPORT
void  vl_sift_calc_keypoint_descriptor   (VlSiftFilt *f,
                                          vl_sift_pix *descr,
                                          VlSiftKeypoint const* k,
                                          double angle) ;

VL_EXPORT
void  vl_sift_calc_raw_descriptor        (VlSiftFilt const *f,
                                          vl_sift_pix const* image,
                                          vl_sift_pix *descr,
                                          int widht, int height,
                                          double x, double y,
                                          double s, double angle0) ;

VL_EXPORT
void  vl_sift_keypoint_init              (VlSiftFilt const *f,
                                          VlSiftKeypoint *k,
                                          double x,
                                          double y,
                                          double sigma) ;





VL_INLINE int    vl_sift_get_octave_index   (VlSiftFilt const *f) ;
VL_INLINE int    vl_sift_get_noctaves       (VlSiftFilt const *f) ;
VL_INLINE int    vl_sift_get_octave_first   (VlSiftFilt const *f) ;
VL_INLINE int    vl_sift_get_octave_width   (VlSiftFilt const *f) ;
VL_INLINE int    vl_sift_get_octave_height  (VlSiftFilt const *f) ;
VL_INLINE int    vl_sift_get_nlevels        (VlSiftFilt const *f) ;
VL_INLINE int    vl_sift_get_nkeypoints     (VlSiftFilt const *f) ;
VL_INLINE double vl_sift_get_peak_thresh    (VlSiftFilt const *f) ;
VL_INLINE double vl_sift_get_edge_thresh    (VlSiftFilt const *f) ;
VL_INLINE double vl_sift_get_norm_thresh    (VlSiftFilt const *f) ;
VL_INLINE double vl_sift_get_magnif         (VlSiftFilt const *f) ;
VL_INLINE double vl_sift_get_window_size    (VlSiftFilt const *f) ;

VL_INLINE vl_sift_pix *vl_sift_get_octave  (VlSiftFilt const *f, int s) ;
VL_INLINE VlSiftKeypoint const *vl_sift_get_keypoints (VlSiftFilt const *f) ;





VL_INLINE void vl_sift_set_peak_thresh (VlSiftFilt *f, double t) ;
VL_INLINE void vl_sift_set_edge_thresh (VlSiftFilt *f, double t) ;
VL_INLINE void vl_sift_set_norm_thresh (VlSiftFilt *f, double t) ;
VL_INLINE void vl_sift_set_magnif      (VlSiftFilt *f, double m) ;
VL_INLINE void vl_sift_set_window_size (VlSiftFilt *f, double m) ;












VL_INLINE int
vl_sift_get_octave_index (VlSiftFilt const *f)
{
  return f-> o_cur ;
}







VL_INLINE int
vl_sift_get_noctaves (VlSiftFilt const *f)
{
  return f-> O ;
}







VL_INLINE int
vl_sift_get_octave_first (VlSiftFilt const *f)
{
  return f-> o_min ;
}







VL_INLINE int
vl_sift_get_octave_width (VlSiftFilt const *f)
{
  return f-> octave_width ;
}







VL_INLINE int
vl_sift_get_octave_height (VlSiftFilt const *f)
{
  return f-> octave_height ;
}













VL_INLINE vl_sift_pix *
vl_sift_get_octave (VlSiftFilt const *f, int s)
{
  int w = vl_sift_get_octave_width  (f) ;
  int h = vl_sift_get_octave_height (f) ;
  return f->octave + w * h * (s - f->s_min) ;
}







VL_INLINE int
vl_sift_get_nlevels (VlSiftFilt const *f)
{
  return f-> S ;
}







VL_INLINE int
vl_sift_get_nkeypoints (VlSiftFilt const *f)
{
  return f-> nkeys ;
}







VL_INLINE VlSiftKeypoint const *
vl_sift_get_keypoints (VlSiftFilt const *f)
{
  return f-> keys ;
}







VL_INLINE double
vl_sift_get_peak_thresh (VlSiftFilt const *f)
{
  return f -> peak_thresh ;
}







VL_INLINE double
vl_sift_get_edge_thresh (VlSiftFilt const *f)
{
  return f -> edge_thresh ;
}







VL_INLINE double
vl_sift_get_norm_thresh (VlSiftFilt const *f)
{
  return f -> norm_thresh ;
}







VL_INLINE double
vl_sift_get_magnif (VlSiftFilt const *f)
{
  return f -> magnif ;
}







VL_INLINE double
vl_sift_get_window_size (VlSiftFilt const *f)
{
  return f -> windowSize ;
}









VL_INLINE void
vl_sift_set_peak_thresh (VlSiftFilt *f, double t)
{
  f -> peak_thresh = t ;
}







VL_INLINE void
vl_sift_set_edge_thresh (VlSiftFilt *f, double t)
{
  f -> edge_thresh = t ;
}







VL_INLINE void
vl_sift_set_norm_thresh (VlSiftFilt *f, double t)
{
  f -> norm_thresh = t ;
}







VL_INLINE void
vl_sift_set_magnif (VlSiftFilt *f, double m)
{
  f -> magnif = m ;
}










VL_INLINE void
vl_sift_set_window_size (VlSiftFilt *f, double x)
{
  f -> windowSize = x ;
}


#endif
