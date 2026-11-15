/*
Copyright (c) 2024, ETH Zurich.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of ETH Zurich nor the names of its contributors may
      be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace tracking {

class PairwiseDirectionResidual {
public:
    explicit PairwiseDirectionResidual(const Eigen::Vector3d& translation_observation)
            : translation_observation_(translation_observation) {}

    template <typename T>
    bool operator()(const T* position1,
                    const T* position2,
                    const T* scale,
                    T* residuals) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_vector(residuals);
        residual_vector =
                translation_observation_.cast<T>() -
                scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1));
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& translation_observation) {
        return new ceres::AutoDiffCostFunction<PairwiseDirectionResidual, 3, 3, 3, 1>(
                new PairwiseDirectionResidual(translation_observation));
    }

    static ceres::CostFunction* Create(const double* translation_observation) {
        const Eigen::Vector3d mapped_observation(
                translation_observation[0], translation_observation[1],
                translation_observation[2]);
        return Create(mapped_observation);
    }

private:
    const Eigen::Vector3d translation_observation_;
};

}  // namespace tracking
