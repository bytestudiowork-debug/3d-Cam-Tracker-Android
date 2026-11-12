// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/optim/support_measurement.h"
namespace colmap {
InlierSupportMeasurer::Support InlierSupportMeasurer::Evaluate(const std::vector<double>& residuals,double max_residual){
 Support support;support.num_inliers=0;support.residual_sum=0;
 for(const auto residual:residuals)if(residual<=max_residual){++support.num_inliers;support.residual_sum+=residual;}
 return support;
}
bool InlierSupportMeasurer::IsLeftBetter(const Support& a,const Support& b){
 if(a.num_inliers>b.num_inliers)return true;return a.num_inliers==b.num_inliers&&a.residual_sum<b.residual_sum;
}
}