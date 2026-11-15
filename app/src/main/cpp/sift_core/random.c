




/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
Copyright (C) 2013 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/
























































#include "random.h"

/*
A C-program for MT19937, with initialization improved 2002/1/26.
Coded by Takuji Nishimura and Makoto Matsumoto.

Before using, initialize the state by using init_genrand(seed)
or init_by_array(init_key, keySize).

Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. The names of its contributors may not be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Any feedback is very welcome.
http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

#include <stdio.h>
#include <string.h>


#define N 624
#define M 397
#define MATRIX_A VL_UINT32_C(0x9908b0df)
#define UPPER_MASK VL_UINT32_C(0x80000000)
#define LOWER_MASK VL_UINT32_C(0x7fffffff)







void
vl_rand_init (VlRand * self)
{
  memset (self->mt, 0, sizeof(self->mt[0]) * N) ;
  self->mti = N + 1 ;
}






void
vl_rand_seed (VlRand * self, vl_uint32 s)
{
#define mti self->mti
#define mt self->mt
  mt[0]= s & VL_UINT32_C(0xffffffff);
  for (mti=1; mti<N; mti++) {
    mt[mti] =
      (VL_UINT32_C(1812433253) * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);




    mt[mti] &= VL_UINT32_C(0xffffffff);

  }
#undef mti
#undef mt
}







void
vl_rand_seed_by_array (VlRand * self, vl_uint32 const key [], vl_size keySize)
{
#define mti self->mti
#define mt self->mt
  int i, j, k;
  vl_rand_seed (self, VL_UINT32_C(19650218));
  i=1; j=0;
  k = (N > keySize ? N : (int)keySize);
  for (; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * VL_UINT32_C(1664525)))
      + key[j] + j;
    mt[i] &= VL_UINT32_C(0xffffffff);
    i++; j++;
    if (i>=N) { mt[0] = mt[N-1]; i=1; }
    if (j>=(signed)keySize) j=0;
  }
  for (k=N-1; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * VL_UINT32_C(1566083941)))
      - i;
    mt[i] &= VL_UINT32_C(0xffffffff) ;
    i++;
    if (i>=N) { mt[0] = mt[N-1]; i=1; }
  }

  mt[0] = VL_UINT32_C(0x80000000);
#undef mti
#undef mt
}









void
vl_rand_permute_indexes (VlRand *self, vl_index *array, vl_size size)
{
  vl_index i, j, tmp;
  for (i = size - 1 ; i > 0; i--) {

    j = (vl_int) vl_rand_uindex (self, i + 1) ;
    tmp = array[i] ; array[i] = array[j] ; array[j] = tmp ;
  }
}







vl_uint32
vl_rand_uint32 (VlRand * self)
{
  vl_uint32 y;
  static vl_uint32 mag01[2]={VL_UINT32_C(0x0), MATRIX_A};


#define mti self->mti
#define mt self->mt

  if (mti >= N) {
    int kk;

    if (mti == N+1)
      vl_rand_seed (self, VL_UINT32_C(5489));

    for (kk=0;kk<N-M;kk++) {
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
      mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & VL_UINT32_C(0x1)];
    }
    for (;kk<N-1;kk++) {
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
      mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & VL_UINT32_C(0x1)];
    }
    y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
    mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & VL_UINT32_C(0x1)];

    mti = 0;
  }

  y = mt[mti++];


  y ^= (y >> 11);
  y ^= (y << 7) & VL_UINT32_C(0x9d2c5680);
  y ^= (y << 15) & VL_UINT32_C(0xefc60000);
  y ^= (y >> 18);

  return (vl_uint32)y;

#undef mti
#undef mt
}
