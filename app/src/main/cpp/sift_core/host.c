





/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/






























































































































































































































































































































































































#include "host.h"
#include "generic.h"
#include <stdio.h>

#if defined(VL_ARCH_IX86) || defined(VL_ARCH_IA64) || defined(VL_ARCH_X64)
#define HAS_CPUID
#else
#undef HAS_CPUID
#endif

#if defined(HAS_CPUID) & defined(VL_COMPILER_MSC)
#include <intrin.h>
VL_INLINE void
_vl_cpuid (vl_int32* info, int function)
{
  __cpuid(info, function) ;
}
#endif

#if defined(HAS_CPUID) & defined(VL_COMPILER_GNUC)
VL_INLINE void
_vl_cpuid (vl_int32* info, int function)
{
#if defined(VL_ARCH_IX86) && (defined(__PIC__) || defined(__pic__))





  __asm__ __volatile__
  ("pushl %%ebx      \n"
   "cpuid            \n"
   "movl %%ebx, %1   \n"
   "popl %%ebx       \n"
   : "=a"(info[0]), "=r"(info[1]), "=c"(info[2]), "=d"(info[3])
   : "a"(function)
   : "cc") ;
#else
  __asm__ __volatile__
  ("cpuid"
   : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
   : "a"(function)
   : "cc") ;
#endif
}

#endif

#if defined(HAS_CPUID)

void
_vl_x86cpu_info_init (VlX86CpuInfo *self)
{
  vl_int32 info [4] ;
  int max_func = 0 ;
  _vl_cpuid(info, 0) ;
  max_func = info[0] ;
  self->vendor.words[0] = info[1] ;
  self->vendor.words[1] = info[3] ;
  self->vendor.words[2] = info[2] ;

  if (max_func >= 1) {
    _vl_cpuid(info, 1) ;
    self->hasMMX   = info[3] & (1 << 23) ;
    self->hasSSE   = info[3] & (1 << 25) ;
    self->hasSSE2  = info[3] & (1 << 26) ;
    self->hasSSE3  = info[2] & (1 <<  0) ;
    self->hasSSE41 = info[2] & (1 << 19) ;
    self->hasSSE42 = info[2] & (1 << 20) ;
    self->hasAVX   = info[2] & (1 << 28) ;
  }
}

#endif

char *
_vl_x86cpu_info_to_string_copy (VlX86CpuInfo const *self)
{
  char * string = 0 ;
  int length = 0 ;
  while (string == 0) {
    if (length > 0) {
      string = vl_malloc(sizeof(char) * length) ;
      if (string == NULL) break ;
    }
    length = snprintf(string, length, "%s%s%s%s%s%s%s%s",
                      self->vendor.string,
                      self->hasMMX   ? " MMX" : "",
                      self->hasSSE   ? " SSE" : "",
                      self->hasSSE2  ? " SSE2" : "",
                      self->hasSSE3  ? " SSE3" : "",
                      self->hasSSE41 ? " SSE41" : "",
                      self->hasSSE42 ? " SSE42" : "",
                      self->hasAVX   ? " AVX" : "") ;
    length += 1 ;
  }
  return string ;
}










VL_EXPORT char *
vl_static_configuration_to_string_copy ()
{
  char const * hostString =
#ifdef VL_ARCH_X64
  "X64"
#endif
#ifdef VL_ARCH_IA64
  "IA64"
#endif
#ifdef VL_ARCH_IX86
  "IX86"
#endif
#ifdef VL_ARCH_PPC
  "PPC"
#endif
  ", "
#ifdef VL_ARCH_BIG_ENDIAN
  "big_endian"
#endif
#ifdef VL_ARCH_LITTLE_ENDIAN
  "little_endian"
#endif
  ;

  char compilerString [1024] ;

  char const * libraryString =
#ifndef VL_DISABLE_THREADS
#ifdef VL_THREADS_WIN
  "Windows_threads"
#elif VL_THREADS_POSIX
  "POSIX_threads"
#endif
#else
  "No_threads"
#endif
#ifndef VL_DISABLE_SSE2
  ", SSE2"
#endif
#if defined(_OPENMP)
  ", OpenMP"
#endif
  ;

snprintf(compilerString, 1024,
#ifdef VL_COMPILER_MSC
  "Microsoft Visual C++ %d"
#define v VL_COMPILER_MSC
#endif
#ifdef VL_COMPILER_GNUC
  "GNU C %d"
#define v VL_COMPILER_GNUC
#endif
  " "
#ifdef VL_COMPILER_LP64
  "LP64"
#endif
#ifdef VL_COMPILER_LLP64
  "LP64"
#endif
#ifdef VL_COMPILER_ILP32
  "ILP32"
#endif
           , v) ;

  {
    char * string = 0 ;
    int length = 0 ;
    while (string == 0) {
      if (length > 0) {
        string = vl_malloc(sizeof(char) * length) ;
        if (string == NULL) break ;
      }
      length = snprintf(string, length, "%s, %s, %s",
                        hostString,
                        compilerString,
                        libraryString) ;
      length += 1 ;
    }
    return string ;
  }
}
