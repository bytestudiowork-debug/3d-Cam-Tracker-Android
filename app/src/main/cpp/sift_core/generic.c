




/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
Copyright (C) 2013 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/




















































































































































































































































































































































































































































































































































































































































































































































































#include "generic.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#if defined(VL_OS_WIN)
#include <Windows.h>
#else
#include <unistd.h>
#endif

#if ! defined(VL_DISABLE_THREADS) && defined(VL_THREADS_POSIX)
#include <pthread.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif






typedef struct _VlThreadState
{

  int lastError ;
  char lastErrorMessage [VL_ERR_MSG_LEN] ;


  VlRand rand ;


#if defined(VL_OS_WIN)
  LARGE_INTEGER ticFreq ;
  LARGE_INTEGER ticMark ;
#else
  clock_t ticMark ;
#endif
} VlThreadState ;


typedef struct _VlState
{


#if defined(VL_DISABLE_THREADS)
  VlThreadState * threadState ;
#else
#if defined(VL_THREADS_POSIX)
  pthread_key_t threadKey ;
  pthread_mutex_t mutex ;
  pthread_t mutexOwner ;
  pthread_cond_t mutexCondition ;
  size_t mutexCount ;
#elif defined(VL_THREADS_WIN)
  DWORD tlsIndex ;
  CRITICAL_SECTION mutex ;
#endif
#endif


  int   (*printf_func)  (char const * format, ...) ;
  void *(*malloc_func)  (size_t) ;
  void *(*realloc_func) (void*,size_t) ;
  void *(*calloc_func)  (size_t, size_t) ;
  void  (*free_func)    (void*) ;

#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  VlX86CpuInfo cpuInfo ;
#endif
  vl_size numCPUs ;
  vl_bool simdEnabled ;
  vl_size numThreads ;
} VlState ;


VlState _vl_state ;


VL_INLINE VlState * vl_get_state () ;
VL_INLINE VlThreadState * vl_get_thread_specific_state () ;
static void vl_lock_state (void) ;
static void vl_unlock_state (void) ;
static VlThreadState * vl_thread_specific_state_new (void) ;
static void vl_thread_specific_state_delete (VlThreadState * self) ;





char const *
vl_get_version_string ()
{
  return VL_VERSION_STRING ;
}








char *
vl_configuration_to_string_copy ()
{
  char * string = 0 ;
  int length = 0 ;
  char * staticString = vl_static_configuration_to_string_copy() ;
  char * cpuString =
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  _vl_x86cpu_info_to_string_copy(&vl_get_state()->cpuInfo) ;
#else
  "Generic CPU" ;
#endif
#if defined(DEBUG)
  int const debug = 1 ;
#else
  int const debug = 0 ;
#endif

  while (string == 0) {
    if (length > 0) {
      string = vl_malloc(sizeof(char) * length) ;
      if (string == NULL) break ;
    }
    length = snprintf(string, length,
                      "Feature runtime version %s\n"
                      "    Static config: %s\n"
                      "    %" VL_FMT_SIZE " CPU(s): %s\n"
#if defined(_OPENMP)
                      "    OpenMP: max threads: %d (library: %" VL_FMT_SIZE ")\n"
#endif
                      "    Debug: %s\n",
                      vl_get_version_string (),
                      staticString,
                      vl_get_num_cpus(), cpuString,
#if defined(_OPENMP)
                      omp_get_max_threads(), vl_get_max_threads(),
#endif
                      VL_YESNO(debug)) ;
    length += 1 ;
  }

  if (staticString) vl_free(staticString) ;
  if (cpuString) vl_free(cpuString) ;
  return string ;
}


static int
do_nothing_printf (char const* format VL_UNUSED, ...)
{
  return 0 ;
}













static void
vl_lock_state (void)
{
#if ! defined(VL_DISABLE_THREADS)
#if   defined(VL_THREADS_POSIX)
  VlState * state = vl_get_state () ;
  pthread_t thisThread = pthread_self () ;
  pthread_mutex_lock (&state->mutex) ;
  if (state->mutexCount >= 1 &&
      pthread_equal (state->mutexOwner, thisThread)) {
    state->mutexCount ++ ;
  } else {
    while (state->mutexCount >= 1) {
      pthread_cond_wait (&state->mutexCondition, &state->mutex) ;
    }
    state->mutexOwner = thisThread ;
    state->mutexCount = 1 ;
  }
  pthread_mutex_unlock (&state->mutex) ;
#elif defined(VL_THREADS_WIN)
  EnterCriticalSection (&vl_get_state()->mutex) ;
#endif
#endif
}









static void
vl_unlock_state (void)
{
#if ! defined(VL_DISABLE_THREADS)
#if   defined(VL_THREADS_POSIX)
  VlState * state = vl_get_state () ;
  pthread_mutex_lock (&state->mutex) ;
  -- state->mutexCount ;
  if (state->mutexCount == 0) {
    pthread_cond_signal (&state->mutexCondition) ;
  }
  pthread_mutex_unlock (&state->mutex) ;
#elif defined(VL_THREADS_WIN)
  LeaveCriticalSection (&vl_get_state()->mutex) ;
#endif
#endif
}









VL_INLINE VlState *
vl_get_state (void)
{
  return &_vl_state ;
}







VL_INLINE VlThreadState *
vl_get_thread_specific_state (void)
{
#ifdef VL_DISABLE_THREADS
  return vl_get_state()->threadState ;
#else
  VlState * state ;
  VlThreadState * threadState ;

  vl_lock_state() ;
  state = vl_get_state() ;

#if defined(VL_THREADS_POSIX)
  threadState = (VlThreadState *) pthread_getspecific(state->threadKey) ;
#elif defined(VL_THREADS_WIN)
  threadState = (VlThreadState *) TlsGetValue(state->tlsIndex) ;
#endif

  if (! threadState) {
    threadState = vl_thread_specific_state_new () ;
  }

#if defined(VL_THREADS_POSIX)
  pthread_setspecific(state->threadKey, threadState) ;
#elif defined(VL_THREADS_WIN)
  TlsSetValue(state->tlsIndex, threadState) ;
#endif

  vl_unlock_state() ;
  return threadState ;
#endif
}






vl_size
vl_get_num_cpus (void)
{
  return vl_get_state()->numCPUs ;
}












void
vl_set_simd_enabled (vl_bool x)
{
  vl_get_state()->simdEnabled = x ;
}





vl_bool
vl_get_simd_enabled (void)
{
  return vl_get_state()->simdEnabled ;
}





vl_bool
vl_cpu_has_avx (void)
{
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  return vl_get_state()->cpuInfo.hasAVX ;
#else
  return VL_FALSE ;
#endif
}





vl_bool
vl_cpu_has_sse3 (void)
{
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  return vl_get_state()->cpuInfo.hasSSE3 ;
#else
  return VL_FALSE ;
#endif
}





vl_bool
vl_cpu_has_sse2 (void)
{
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  return vl_get_state()->cpuInfo.hasSSE2 ;
#else
  return VL_FALSE ;
#endif
}














vl_size
vl_get_thread_limit (void)
{
#if defined(_OPENMP)
#if _OPENMP >= 200805

  return omp_get_thread_limit() ;
#else
  return 0 ;
#endif
#else
  return 1 ;
#endif
}


















vl_size
vl_get_max_threads (void)
{
#if defined(_OPENMP)
  return vl_get_state()->numThreads ;
#else
  return 1 ;
#endif
}























#if defined(_OPENMP)
void
vl_set_num_threads (vl_size numThreads)
{
  if (numThreads == 0) {
    numThreads = omp_get_max_threads() ;
  }
  vl_get_state()->numThreads = numThreads ;
}
#else
void
vl_set_num_threads (vl_size numThreads VL_UNUSED) { }
#endif


















int
vl_set_last_error (int error, char const * errorMessage, ...)
{
  VlThreadState * state = vl_get_thread_specific_state() ;
  va_list args;
  va_start(args, errorMessage) ;
  if (errorMessage) {
#ifdef VL_COMPILER_LCC
    vsprintf(state->lastErrorMessage, errorMessage, args) ;
#else
    vsnprintf(state->lastErrorMessage,
              sizeof(state->lastErrorMessage)/sizeof(char),
              errorMessage, args) ;
#endif
  } else {
    state->lastErrorMessage[0] = 0 ;
  }
  state->lastError = error ;
  va_end(args) ;
  return error ;
}






int
vl_get_last_error (void) {
  return vl_get_thread_specific_state()->lastError ;
}






char const *
vl_get_last_error_message (void)
{
  return vl_get_thread_specific_state()->lastErrorMessage ;
}









void
vl_set_alloc_func (void *(*malloc_func)  (size_t),
                   void *(*realloc_func) (void*, size_t),
                   void *(*calloc_func)  (size_t, size_t),
                   void  (*free_func)    (void*))
{
  VlState * state ;
  vl_lock_state () ;
  state = vl_get_state() ;
  state->malloc_func  = malloc_func ;
  state->realloc_func = realloc_func ;
  state->calloc_func  = calloc_func ;
  state->free_func    = free_func ;
  vl_unlock_state () ;
}









void *
vl_malloc (size_t n)
{
  return (vl_get_state()->malloc_func)(n) ;

}











void *
vl_realloc (void* ptr, size_t n)
{
  return (vl_get_state()->realloc_func)(ptr, n) ;
}










void *
vl_calloc (size_t n, size_t size)
{
  return (vl_get_state()->calloc_func)(n, size) ;
}









void
vl_free (void *ptr)
{
  (vl_get_state()->free_func)(ptr) ;
}








void
vl_set_printf_func (printf_func_t printf_func)
{
  vl_get_state()->printf_func = printf_func ? printf_func : do_nothing_printf ;
}






printf_func_t
vl_get_printf_func (void) {
  return vl_get_state()->printf_func ;
}







double
vl_get_cpu_time ()
{
  #ifdef VL_OS_WIN
  VlThreadState * threadState = vl_get_thread_specific_state() ;
  LARGE_INTEGER mark ;
  QueryPerformanceCounter (&mark) ;
  return (double)mark.QuadPart / (double)threadState->ticFreq.QuadPart ;
#else
  return (double)clock() / (double)CLOCKS_PER_SEC ;
#endif
}







void
vl_tic (void)
{
  VlThreadState * threadState = vl_get_thread_specific_state() ;
#ifdef VL_OS_WIN
  QueryPerformanceCounter (&threadState->ticMark) ;
#else
  threadState->ticMark = clock() ;
#endif
}















double
vl_toc (void)
{
  VlThreadState * threadState = vl_get_thread_specific_state() ;
#ifdef VL_OS_WIN
  LARGE_INTEGER tocMark ;
  QueryPerformanceCounter(&tocMark) ;
  return (double) (tocMark.QuadPart - threadState->ticMark.QuadPart) /
    threadState->ticFreq.QuadPart ;
#else
  return (double) (clock() - threadState->ticMark) / CLOCKS_PER_SEC ;
#endif
}










VL_EXPORT VlRand *
vl_get_rand (void)
{
  return &vl_get_thread_specific_state()->rand ;
}









static VlThreadState *
vl_thread_specific_state_new (void)
{
  VlThreadState * self ;
#if defined(DEBUG)
  printf("Feature runtime DEBUG: thread constructor begins.\n") ;
#endif
  self = malloc(sizeof(VlThreadState)) ;
  self->lastError = 0 ;
  self->lastErrorMessage[0] = 0 ;
#if defined(VL_OS_WIN)
  QueryPerformanceFrequency (&self->ticFreq) ;
  self->ticMark.QuadPart = 0 ;
#else
  self->ticMark = 0 ;
#endif
  vl_rand_init (&self->rand) ;

  return self ;
}





static void
vl_thread_specific_state_delete (VlThreadState * self)
{
#if defined(DEBUG)
  printf("Feature runtime DEBUG: thread destructor begins.\n") ;
#endif
  free (self) ;
}






void
vl_constructor (void)
{
  VlState * state ;
#if defined(DEBUG)
  printf("Feature runtime DEBUG: constructor begins.\n") ;
#endif

  state = vl_get_state() ;

#if ! defined(VL_DISABLE_THREADS)
#if defined(DEBUG)
  printf("Feature runtime DEBUG: constructing thread specific state.\n") ;
#endif
#if defined(VL_THREADS_POSIX)
  {
    typedef void (*destructorType)(void * );
    pthread_key_create (&state->threadKey,
                        (destructorType)
                          vl_thread_specific_state_delete) ;
    pthread_mutex_init (&state->mutex, NULL) ;
    pthread_cond_init (&state->mutexCondition, NULL) ;
  }
#elif defined(VL_THREADS_WIN)
  InitializeCriticalSection (&state->mutex) ;
  state->tlsIndex = TlsAlloc () ;
#endif
#else


#if defined(DEBUG)
  printf("Feature runtime DEBUG: constructing the generic thread state instance (threading support disabled).\n") ;
#endif
  vl_get_state()->threadState = vl_thread_specific_state_new() ;
#endif

  state->malloc_func  = malloc ;
  state->realloc_func = realloc ;
  state->calloc_func  = calloc ;
  state->free_func    = free ;
  state->printf_func  = printf ;


#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  _vl_x86cpu_info_init (&state->cpuInfo) ;
#endif


#if defined(VL_OS_WIN)
  {
    SYSTEM_INFO info;
    GetSystemInfo (&info) ;
    state->numCPUs = info.dwNumberOfProcessors ;
  }
#elif defined(_SC_NPROCESSORS_ONLN)
  state->numCPUs = sysconf(_SC_NPROCESSORS_ONLN) ;
#else
  state->numCPUs = 1 ;
#endif
  state->simdEnabled = VL_TRUE ;


#if defined(_OPENMP)
  state->numThreads = omp_get_max_threads() ;
#else
  state->numThreads = 1 ;
#endif

#if defined(DEBUG)
  printf("Feature runtime DEBUG: constructor ends.\n") ;
#endif
}


void
vl_destructor ()
{
  VlState * state ;
#if defined(DEBUG)
  printf("Feature runtime DEBUG: destructor begins.\n") ;
#endif

  state = vl_get_state() ;

#if ! defined(VL_DISABLE_THREADS)
#if defined(DEBUG)
  printf("Feature runtime DEBUG: destroying a thread specific state instance.\n") ;
#endif
#if   defined(VL_THREADS_POSIX)
  {






    VlThreadState * threadState =
       pthread_getspecific(state->threadKey) ;
    if (threadState) {
      vl_thread_specific_state_delete (threadState) ;
      pthread_setspecific(state->threadKey, NULL) ;
    }
  }
  pthread_cond_destroy (&state->mutexCondition) ;
  pthread_mutex_destroy (&state->mutex) ;
  pthread_key_delete (state->threadKey) ;
#elif defined(VL_THREADS_WIN)
 {






    VlThreadState * threadState =
       TlsGetValue(state->tlsIndex) ;
    if (threadState) {
      vl_thread_specific_state_delete (threadState) ;
      TlsSetValue(state->tlsIndex, NULL) ;
    }
  }
  TlsFree (state->tlsIndex) ;
  DeleteCriticalSection (&state->mutex) ;
#endif
#else
#if defined(DEBUG)
  printf("Feature runtime DEBUG: destroying the generic thread state instance (threading support disabled).\n") ;
#endif
  vl_thread_specific_state_delete(vl_get_state()->threadState) ;
#endif

#if defined(DEBUG)
  printf("Feature runtime DEBUG: destructor ends.\n") ;
#endif
}





#ifdef __cplusplus
    #define INITIALIZER(f) \
        static void f(void); \
        struct f##_t_ { f##_t_(void) { f(); } }; static f##_t_ f##_; \
        static void f(void)
#elif defined(_MSC_VER)
    #pragma section(".CRT$XCU",read)
    #define INITIALIZER2_(f,p) \
        static void f(void); \
        __declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
        __pragma(comment(linker,"/include:" p #f "_")) \
        static void f(void)
    #ifdef _WIN64
        #define INITIALIZER(f) INITIALIZER2_(f,"")
    #else
        #define INITIALIZER(f) INITIALIZER2_(f,"_")
    #endif
#else
    #define INITIALIZER(f) \
        static void f(void) __attribute__((constructor)); \
        static void f(void)
#endif

INITIALIZER(vl_initialize)
{
    vl_constructor();
    atexit(vl_destructor);
}
