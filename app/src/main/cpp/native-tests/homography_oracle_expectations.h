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

#include <cstddef>

namespace homography_oracle_expectations {

// Fixed outputs emitted by an independently compiled build at the revision
// recorded in the two-view geometry license notice.
constexpr double kMinimalDlt[9] = {
        10.124167797030394, -27.5850494863523, 1471.3081260872195,
        2.9427681683397706, -9.7287649455457963, 816.3447206153262,
        0.02504329342214523, -0.053272822745883915, 1.0};
constexpr double kOverdeterminedDlt[9] = {
        0.0032920572880451747, -6.3611027849041498e-05, 0.96807070995296629,
        4.6998601803864683e-05, 0.0038015579828934542, -0.25059510865166612,
        -5.9790526760288022e-07, 2.8912350821919217e-07,
        0.0039870178156396754};
constexpr double kFirstTwelveOneWayResiduals[12] = {
        5.7206153958517621e-05, 0.0011416112217847293, 0.0062857799358108767,
        0.0019086232154573535, 0.034045475815995901, 0.014769118619808164,
        0.017172112155351305, 0.021761415375573107, 0.013698335998070342,
        0.006755433687614068, 0.0033316183502866675, 0.0079424704489886901};

constexpr std::size_t kPlanarTrials = 53;
constexpr std::size_t kPlanarInliers = 40;
constexpr double kPlanarResidualSum = 0.48335517208992879;
constexpr char kPlanarMask[] =
        "1111111111111111111111111111111111111111000000000000";
constexpr double kPlanarRansacModel[9] = {
        0.0032951367174502222, -6.6365613526106268e-05,
        0.96813371827437444, 4.4897696520455421e-05,
        0.0038050375216852552, -0.25035147102823607,
        -5.9550404297530942e-07, 2.8659021105180771e-07,
        0.0039877107556117921};

constexpr double kPlanarPoseRotations[2][9] = {
        {0.98867320228358924, -0.064915093964585585, 0.13531936174059525,
         0.072471299234604353, 0.9960307833408647, -0.051677745932189365,
         -0.13142758414260489, 0.060899172514513739, 0.98945342533819447},
        {0.97394043735419844, -0.061653657954823131, 0.2182632606443472,
         0.076985116235824713, 0.99507498475523093, -0.062442506296059276,
         -0.21333850183328168, 0.077618304382191444, 0.97389017987674786}};
constexpr double kPlanarPoseTranslations[4][3] = {
        {0.087434620953058442, -0.012118329005913293, 0.0086785247176427074},
        {-0.087434620953058442, 0.012118329005913293, -0.0086785247176427074},
        {0.01612712952169933, -0.004861707004248348, 0.087081933423474617},
        {-0.01612712952169933, 0.004861707004248348, -0.087081933423474617}};
constexpr std::size_t kPlanarPositiveDepthCounts[4] = {40, 0, 40, 0};
constexpr double kPlanarPoseResidualSums[4] = {
        8.6755437611607533e-08, 0.0, 9.7590864167607094e-08, 0.0};
constexpr std::size_t kPlanarSelectedPose = 0;

constexpr std::size_t kNonPlanarTrials = 180;
constexpr std::size_t kNonPlanarInliers = 21;
constexpr double kNonPlanarResidualSum = 29.51009489377676;
constexpr char kNonPlanarMask[] =
        "111111111000000000000011100000000000111111111000000000000000";
constexpr double kNonPlanarRansacModel[9] = {
        0.002464336306434659, -0.00075388486380035017,
        0.98151255833580897, 3.9177724311961857e-05,
        0.00285741920247905, -0.1913337802202103,
        -4.426655968545908e-07, -1.2337503082131105e-07,
        0.0031104180291608541};
constexpr double kNonPlanarPoseRotations[2][9] = {
        {0.98491482494561833, -0.08440628260180344, 0.15105749587387754,
         0.089649769041329291, 0.99557311820458749, -0.028232697696654929,
         -0.14800576513493771, 0.041349072126698125, 0.98812172717792357},
        {0.96159477202920252, -0.20159205611309, 0.18626899183224976,
         0.19886307924421365, 0.9794572878804465, 0.033420007950175926,
         -0.18917972967423219, 0.0049055203574129239, 0.98193022448156031}};
constexpr double kNonPlanarPoseTranslations[4][3] = {
        {-0.17560580147027571, 0.084331442727792486, -0.087344703219187747},
        {0.17560580147027571, -0.084331442727792486, 0.087344703219187747},
        {-0.07653336912194536, 0.1817346603033225, -0.081813468196238123},
        {0.07653336912194536, -0.1817346603033225, 0.081813468196238123}};
constexpr std::size_t kNonPlanarPositiveDepthCounts[4] = {0, 21, 0, 21};
constexpr double kNonPlanarPoseResidualSums[4] = {
        0.0, 2.8874745822138337e-06, 0.0, 6.3246193777732884e-06};
constexpr std::size_t kNonPlanarSelectedPose = 1;
constexpr double kNonPlanarMedianAngle = 0.078026021976537518;

constexpr double kLowParallaxPoseRotations[2][9] = {
        {0.99972517580668707, -0.010670309260509226, 0.020873843885918775,
         0.010851075009235116, 0.9999044351741736, -0.0085659027639136548,
         -0.020780448249077082, 0.0087900522923337332, 0.99974542157054436},
        {0.99949803967444917, -0.010607167367341771, 0.029852247610095382,
         0.01093849532251478, 0.99988013307821288, -0.010957590774185713,
         -0.029732440313781305, 0.01127862916920303, 0.99949425937173397}};
constexpr double kLowParallaxPoseTranslations[4][3] = {
        {0.0089999999999940562, -0.0023999999999983696,
         0.00060000000000002272},
        {-0.0089999999999940562, 0.0023999999999983696,
         -0.00060000000000002272},
        {0.000236736394265999, -9.1115268836884514e-05,
         0.0093303619376367136},
        {-0.000236736394265999, 9.1115268836884514e-05,
         -0.0093303619376367136}};
constexpr std::size_t kLowParallaxPositiveDepthCounts[4] = {32, 0, 14, 18};
constexpr double kLowParallaxPoseResidualSums[4] = {
        2.4424906541753444e-15, 0.0, 1.4432899320127035e-15,
        1.3322676295501878e-15};
constexpr std::size_t kLowParallaxSelectedPose = 0;
constexpr double kLowParallaxMedianAngle = 0.0091271784667569921;

constexpr std::size_t kWideParallaxInliers = 120;
constexpr std::size_t kWideParallaxTrials = 22;
constexpr double kWideParallaxResidualSum = 6.0628056332054665e-23;
constexpr double kWideParallaxRansacModel[9] = {
        0.82724058227964292, -0.017571938912298569, 1081.9572020633934,
        0.011377184960028557, 0.95480868738177849, -67.181374909119185,
        -0.00014885843136046403, 7.0729180606427891e-05, 1.0};
constexpr double kWideParallaxMedianAngle = 0.2857582127715716;
constexpr double kWideParallaxSelectedTranslation[3] = {
        1.1851851851851849, -0.018518518518517095,
        0.0074074074074075569};

}  // namespace homography_oracle_expectations
