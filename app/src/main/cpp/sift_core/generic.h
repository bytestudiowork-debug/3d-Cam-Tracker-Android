




/*
Copyright (C) 2013 Andrea Vedaldi.
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_GENERIC_H
#define VL_GENERIC_H

#include "host.h"
#include "random.h"

#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <assert.h>


#define VL_VERSION_STRING "0.9.20"


#define VL_ERR_MSG_LEN 1024




#define VL_TYPE_FLOAT   1
#define VL_TYPE_DOUBLE  2
#define VL_TYPE_INT8    3
#define VL_TYPE_UINT8   4
#define VL_TYPE_INT16   5
#define VL_TYPE_UINT16  6
#define VL_TYPE_INT32   7
#define VL_TYPE_UINT32  8
#define VL_TYPE_INT64   9
#define VL_TYPE_UINT64  10

typedef vl_uint32 vl_type ;










void vl_constructor();
void vl_destructor();

VL_INLINE char const *
vl_get_type_name (vl_type type)
{
  switch (type) {
    case VL_TYPE_FLOAT   : return "float"  ;
    case VL_TYPE_DOUBLE  : return "double" ;
    case VL_TYPE_INT8    : return "int8"   ;
    case VL_TYPE_INT16   : return "int16"  ;
    case VL_TYPE_INT32   : return "int32"  ;
    case VL_TYPE_INT64   : return "int64"  ;
    case VL_TYPE_UINT8   : return "int8"   ;
    case VL_TYPE_UINT16  : return "int16"  ;
    case VL_TYPE_UINT32  : return "int32"  ;
    case VL_TYPE_UINT64  : return "int64"  ;
    default: return NULL ;
  }
}










VL_INLINE vl_size
vl_get_type_size (vl_type type)
{
  vl_size dataSize = 0 ;
  switch (type) {
    case VL_TYPE_DOUBLE : dataSize = sizeof(double) ; break ;
    case VL_TYPE_FLOAT  : dataSize = sizeof(float) ; break ;
    case VL_TYPE_INT64  : case VL_TYPE_UINT64 : dataSize = sizeof(vl_int64) ; break ;
    case VL_TYPE_INT32  : case VL_TYPE_UINT32 : dataSize = sizeof(vl_int32) ; break ;
    case VL_TYPE_INT16  : case VL_TYPE_UINT16 : dataSize = sizeof(vl_int16) ; break ;
    case VL_TYPE_INT8   : case VL_TYPE_UINT8  : dataSize = sizeof(vl_int8)  ; break ;
    default:
      abort() ;
  }
  return dataSize ;
}


VL_EXPORT char const * vl_get_version_string (void) ;
VL_EXPORT char * vl_configuration_to_string_copy (void) ;
VL_EXPORT void vl_set_simd_enabled (vl_bool x) ;
VL_EXPORT vl_bool vl_get_simd_enabled (void) ;
VL_EXPORT vl_bool vl_cpu_has_avx (void) ;
VL_EXPORT vl_bool vl_cpu_has_sse3 (void) ;
VL_EXPORT vl_bool vl_cpu_has_sse2 (void) ;
VL_EXPORT vl_size vl_get_num_cpus (void) ;
VL_EXPORT VlRand * vl_get_rand (void) ;



VL_EXPORT vl_size vl_get_max_threads (void) ;
VL_EXPORT void vl_set_num_threads (vl_size n) ;
VL_EXPORT vl_size vl_get_thread_limit (void) ;





#define VL_ERR_OK       0
#define VL_ERR_OVERFLOW 1
#define VL_ERR_ALLOC    2
#define VL_ERR_BAD_ARG  3
#define VL_ERR_IO       4
#define VL_ERR_EOF      5
#define VL_ERR_NO_MORE  5

VL_EXPORT int vl_get_last_error (void) ;
VL_EXPORT char const *  vl_get_last_error_message (void) ;
VL_EXPORT int vl_set_last_error (int error, char const * errorMessage, ...) ;





VL_EXPORT void
vl_set_alloc_func (void *(*malloc_func)  (size_t),
                   void *(*realloc_func) (void*,size_t),
                   void *(*calloc_func)  (size_t, size_t),
                   void  (*free_func)    (void*)) ;
VL_EXPORT void *vl_malloc (size_t n) ;
VL_EXPORT void *vl_realloc (void *ptr, size_t n) ;
VL_EXPORT void *vl_calloc (size_t n, size_t size) ;
VL_EXPORT void *vl_memalign (size_t n, size_t size) ;
VL_EXPORT void  vl_free (void* ptr) ;






typedef int(*printf_func_t) (char const *format, ...) ;
VL_EXPORT void vl_set_printf_func (printf_func_t printf_func) ;
VL_EXPORT printf_func_t vl_get_printf_func (void) ;











#define VL_PRINTF (*vl_get_printf_func())
#define VL_PRINT (*vl_get_printf_func())











#define VL_MIN(x,y) (((x)<(y))?(x):(y))






#define VL_MAX(x,y) (((x)>(y))?(x):(y))








#define VL_SHIFT_LEFT(x,n) (((n)>=0)?((x)<<(n)):((x)>>-(n)))






VL_EXPORT void vl_tic (void) ;
VL_EXPORT double vl_toc (void) ;
VL_EXPORT double vl_get_cpu_time (void) ;



#endif
