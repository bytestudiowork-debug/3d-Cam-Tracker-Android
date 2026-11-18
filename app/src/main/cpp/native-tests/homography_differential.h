// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "../homography_estimator_core.h"

namespace tracking::homography_test {
using tracking::source_homography::CalculateTriangulationAngle;
using tracking::source_homography::DecomposeHomography;
using tracking::source_homography::EstimateDLT;
using tracking::source_homography::HomographyPoseCandidate;
using tracking::source_homography::HomographyPoseReport;
using tracking::source_homography::InitialPairGates;
using tracking::source_homography::LORansacReport;
using tracking::source_homography::Matrix;
using tracking::source_homography::Median;
using tracking::source_homography::OneWaySquaredReprojectionResiduals;
using tracking::source_homography::PassInitialPairGates;
using tracking::source_homography::Point;
using tracking::source_homography::Ray;
using tracking::source_homography::RecoverHomographyPose;
using tracking::source_homography::RansacOptions;
using tracking::source_homography::RunHomographyLORansac;
using tracking::source_homography::RunLORansac;
}  // namespace tracking::homography_test
