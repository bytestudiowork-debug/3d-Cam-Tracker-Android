// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
namespace colmap {
inline double DegToRad(double d) { return d * 0.0174532925199432954743716805978692718781530857086181640625; }
namespace internal {
template<class I> bool NextCombination(I first1,I last1,I first2,I last2) {
 if(first1==last1||first2==last2)return false;
 I m1=last1,m2=last2;--m1;--m2;
 while(m1!=first1&&*m1>=*m2)--m1;
 bool result=(m1==first1)&&*first1>=*m2;
 if(!result){while(first2!=m2&&*m1>=*first2)++first2;first1=m1;std::iter_swap(first1,first2);++first1;++first2;}
 if(first1!=last1&&first2!=last2){m1=last1;m2=first2;while(m1!=first1&&m2!=last2){std::iter_swap(--m1,m2);++m2;}std::reverse(first1,m1);std::reverse(first1,last1);std::reverse(m2,last2);std::reverse(first2,last2);}
 return !result;
}}
template<class I> bool NextCombination(I first,I middle,I last){return internal::NextCombination(first,middle,middle,last);}
inline uint64_t NChooseK(uint64_t n,uint64_t k){
 if(n==0||n<k)return 0;k=std::min(k,n-k);uint64_t r=1;
 for(uint64_t d=1;d<=k;++d){const uint64_t divisor=std::gcd(n,d);r/=d/divisor;r*=n--/divisor;}return r;
}
}