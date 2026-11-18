#include <jni.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <vector>

#include <Eigen/Core>

#include "two_view_epipolar_estimator_core.h"
#include "two_view_geometry_core.h"

namespace {

using tracking::source_two_view::EpipolarGeometryReport;
using tracking::source_two_view::EssentialPoseResult;
using tracking::source_two_view::EstimateInitialPairGeometry;
using tracking::source_two_view::EstimateTwoViewEpipolarGeometry;
using tracking::source_two_view::InitialPairGeometryReport;
using tracking::source_two_view::InitialPairPoseResult;
using tracking::source_two_view::RecoverEssentialPoseFromPixels;
using tracking::source_two_view::RansacOptions;
using tracking::source_two_view::RecoverInitialPairPose;
using tracking::source_two_view::TwoViewDispatchOptions;

constexpr std::size_t kModelHeaderSize = 17;
constexpr std::size_t kPoseHeaderSize = 15;
constexpr std::size_t kInitialPairHeaderSize = 41;
constexpr std::size_t kMaxArrayValues =
        tracking::source_two_view::kMaxAndroidInputObservations * 2;

bool ReadDoubleArray(JNIEnv* env,
                     jdoubleArray array,
                     const std::size_t expected_size,
                     std::vector<double>* values) {
    if (array == nullptr || values == nullptr ||
        static_cast<std::size_t>(env->GetArrayLength(array)) != expected_size) {
        return false;
    }
    values->resize(expected_size);
    env->GetDoubleArrayRegion(array, 0, static_cast<jsize>(expected_size),
                              values->data());
    return !env->ExceptionCheck();
}

bool ReadPixelPairs(JNIEnv* env,
                    jdoubleArray first,
                    jdoubleArray second,
                    std::vector<Eigen::Vector2d>* pixels1,
                    std::vector<Eigen::Vector2d>* pixels2) {
    if (first == nullptr || second == nullptr || pixels1 == nullptr ||
        pixels2 == nullptr) {
        return false;
    }
    const jsize first_size = env->GetArrayLength(first);
    const jsize second_size = env->GetArrayLength(second);
    if (first_size != second_size || first_size < 14 ||
        first_size > static_cast<jsize>(kMaxArrayValues) || first_size % 2 != 0) {
        return false;
    }
    std::vector<double> flat1;
    std::vector<double> flat2;
    if (!ReadDoubleArray(env, first, static_cast<std::size_t>(first_size), &flat1) ||
        !ReadDoubleArray(env, second, static_cast<std::size_t>(second_size), &flat2)) {
        return false;
    }
    const std::size_t count = flat1.size() / 2;
    pixels1->resize(count);
    pixels2->resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        (*pixels1)[i] = Eigen::Vector2d(flat1[2 * i], flat1[2 * i + 1]);
        (*pixels2)[i] = Eigen::Vector2d(flat2[2 * i], flat2[2 * i + 1]);
    }
    return true;
}

bool ReadCameraMatrix(JNIEnv* env,
                      jdoubleArray camera_values,
                      Eigen::Matrix3d* camera_matrix) {
    std::vector<double> values;
    if (!ReadDoubleArray(env, camera_values, 9, &values)) {
        return false;
    }
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            (*camera_matrix)(row, col) = values[static_cast<std::size_t>(row * 3 + col)];
        }
    }
    return true;
}

void PackModel(const tracking::source_two_view::LORansacReport<Eigen::Matrix3d>&
                       report,
               const std::size_t count,
               std::vector<double>* packed,
               const std::size_t block_start) {
    (*packed)[block_start] = 1.0;
    (*packed)[block_start + 1] = report.success ? 1.0 : 0.0;
    (*packed)[block_start + 2] = report.success
            ? static_cast<double>(report.support.num_inliers)
            : 0.0;
    (*packed)[block_start + 3] = static_cast<double>(report.num_trials);
    (*packed)[block_start + 4] = report.success ? report.support.residual_sum : 0.0;
    (*packed)[block_start + 5] =
            static_cast<double>(report.local_optimization_trials);
    (*packed)[block_start + 6] =
            static_cast<double>(report.accepted_local_refits);
    (*packed)[block_start + 7] = report.best_model_is_local ? 1.0 : 0.0;
    if (report.success && report.model.has_value()) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                (*packed)[block_start + 8 + static_cast<std::size_t>(row * 3 + col)] =
                        (*report.model)(row, col);
            }
        }
        if (report.inlier_mask.size() == count) {
            for (std::size_t i = 0; i < count; ++i) {
                (*packed)[block_start + kModelHeaderSize + i] =
                        report.inlier_mask[i] != 0 ? 1.0 : 0.0;
            }
        }
    }
}

jdoubleArray MakeDoubleArray(JNIEnv* env, const std::vector<double>& values) {
    jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(values.size()));
    if (result == nullptr) {
        return nullptr;
    }
    env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(values.size()),
                              values.data());
    if (env->ExceptionCheck()) {
        env->DeleteLocalRef(result);
        return nullptr;
    }
    return result;
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_SourceFaithfulTwoViewEstimator_nativeEstimate(
        JNIEnv* env,
        jobject,
        jdoubleArray first_pixels_xy,
        jdoubleArray second_pixels_xy,
        jdoubleArray camera_matrix_values,
        jdouble max_error_pixels) {
    try {
        std::vector<Eigen::Vector2d> pixels1;
        std::vector<Eigen::Vector2d> pixels2;
        Eigen::Matrix3d camera_matrix;
        if (!ReadPixelPairs(env, first_pixels_xy, second_pixels_xy,
                            &pixels1, &pixels2) ||
            !ReadCameraMatrix(env, camera_matrix_values, &camera_matrix)) {
            return nullptr;
        }
        EpipolarGeometryReport report;
        RansacOptions options;
        options.max_num_trials = tracking::source_two_view::kAndroidTrialLimit;
        options.random_seed = 20261118;
        if (!EstimateTwoViewEpipolarGeometry(
                    pixels1, pixels2, camera_matrix, max_error_pixels,
                    &report, options)) {
            return nullptr;
        }

        const std::size_t count = pixels1.size();
        const std::size_t block_size = kModelHeaderSize + count;
        std::vector<double> packed(2 * block_size, 0.0);
        PackModel(report.essential, count, &packed, 0);
        PackModel(report.fundamental, count, &packed, block_size);
        return MakeDoubleArray(env, packed);
    } catch (const std::exception&) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_SourceFaithfulTwoViewEstimator_nativeRecoverPose(
        JNIEnv* env,
        jobject,
        jdoubleArray essential_values,
        jdoubleArray first_pixels_xy,
        jdoubleArray second_pixels_xy,
        jdoubleArray camera_matrix_values,
        jintArray epipolar_inlier_mask) {
    try {
        std::vector<Eigen::Vector2d> pixels1;
        std::vector<Eigen::Vector2d> pixels2;
        Eigen::Matrix3d camera_matrix;
        Eigen::Matrix3d essential;
        if (!ReadPixelPairs(env, first_pixels_xy, second_pixels_xy,
                            &pixels1, &pixels2) ||
            !ReadCameraMatrix(env, camera_matrix_values, &camera_matrix) ||
            !ReadCameraMatrix(env, essential_values, &essential) ||
            epipolar_inlier_mask == nullptr ||
            static_cast<std::size_t>(env->GetArrayLength(epipolar_inlier_mask)) !=
                    pixels1.size()) {
            return nullptr;
        }
        std::vector<jint> mask_values(pixels1.size());
        env->GetIntArrayRegion(epipolar_inlier_mask, 0,
                               static_cast<jsize>(mask_values.size()),
                               mask_values.data());
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        std::vector<char> mask(mask_values.size(), 0);
        for (std::size_t i = 0; i < mask_values.size(); ++i) {
            if (mask_values[i] != 0 && mask_values[i] != 1) {
                return nullptr;
            }
            mask[i] = mask_values[i] == 0 ? 0 : 1;
        }

        EssentialPoseResult pose;
        if (!RecoverEssentialPoseFromPixels(
                    essential, pixels1, pixels2, camera_matrix, mask, &pose)) {
            return nullptr;
        }
        std::vector<double> packed(kPoseHeaderSize + pixels1.size(), 0.0);
        packed[0] = 1.0;
        packed[1] = static_cast<double>(pose.positive_depth_count);
        packed[2] = static_cast<double>(pose.selected_candidate);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                packed[3 + static_cast<std::size_t>(row * 3 + col)] =
                        pose.rotation(row, col);
            }
        }
        for (int i = 0; i < 3; ++i) {
            packed[12 + static_cast<std::size_t>(i)] = pose.translation(i);
        }
        for (std::size_t i = 0; i < pose.cheirality_mask.size(); ++i) {
            packed[kPoseHeaderSize + i] =
                    pose.cheirality_mask[i] != 0 ? 1.0 : 0.0;
        }
        return MakeDoubleArray(env, packed);
    } catch (const std::exception&) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_SourceFaithfulTwoViewEstimator_nativeEstimateInitialPair(
        JNIEnv* env,
        jobject,
        jdoubleArray first_pixels_xy,
        jdoubleArray second_pixels_xy,
        jdoubleArray camera_matrix_values,
        jboolean both_cameras_have_focal_priors,
        jdouble max_error_pixels,
        jint minimum_inliers,
        jdouble minimum_essential_to_fundamental_ratio,
        jdouble maximum_homography_ratio) {
    try {
        if ((both_cameras_have_focal_priors != JNI_TRUE &&
             both_cameras_have_focal_priors != JNI_FALSE) ||
            minimum_inliers <= 0) {
            return nullptr;
        }
        std::vector<Eigen::Vector2d> pixels1;
        std::vector<Eigen::Vector2d> pixels2;
        Eigen::Matrix3d camera_matrix;
        if (!ReadPixelPairs(env, first_pixels_xy, second_pixels_xy,
                            &pixels1, &pixels2) ||
            !ReadCameraMatrix(env, camera_matrix_values, &camera_matrix)) {
            return nullptr;
        }
        RansacOptions epipolar_options;
        epipolar_options.max_num_trials =
                tracking::source_two_view::kAndroidTrialLimit;
        epipolar_options.random_seed = 20261118;
        RansacOptions homography_options;
        homography_options.max_num_trials =
                tracking::source_two_view::kAndroidTrialLimit;
        homography_options.random_seed = 20261118;
        TwoViewDispatchOptions dispatch_options;
        dispatch_options.minimum_inliers =
                static_cast<std::size_t>(minimum_inliers);
        dispatch_options.minimum_essential_to_fundamental_ratio =
                minimum_essential_to_fundamental_ratio;
        dispatch_options.maximum_homography_ratio = maximum_homography_ratio;
        InitialPairGeometryReport geometry;
        if (!EstimateInitialPairGeometry(
                    pixels1, pixels2, camera_matrix,
                    both_cameras_have_focal_priors == JNI_TRUE,
                    max_error_pixels, &geometry, epipolar_options,
                    homography_options, dispatch_options)) {
            return nullptr;
        }
        InitialPairPoseResult pose;
        const bool pose_recovered = geometry.decision.pose_source !=
                        tracking::source_two_view::PoseSource::kNone &&
                RecoverInitialPairPose(
                        geometry, pixels1, pixels2, camera_matrix,
                        tracking::source_homography::InitialPairGates{}, &pose);
        const std::size_t count = pixels1.size();
        std::vector<double> packed(kInitialPairHeaderSize + 5 * count, 0.0);
        packed[0] = 1.0;
        packed[1] = static_cast<double>(geometry.decision.configuration);
        packed[2] = static_cast<double>(geometry.decision.selected_model);
        packed[3] = static_cast<double>(geometry.decision.pose_source);
        packed[4] = static_cast<double>(geometry.decision.num_inliers);
        packed[5] = pose_recovered && pose.success ? 1.0 : 0.0;
        packed[6] = static_cast<double>(pose.selected_candidate);
        packed[7] = static_cast<double>(pose.positive_depth_count);
        packed[8] = pose.median_triangulation_angle_radians;
        packed[9] = pose.passes_initial_pair_gates ? 1.0 : 0.0;
        const auto pack_report = [&packed](
                const tracking::source_two_view::LORansacReport<Eigen::Matrix3d>&
                        report,
                const std::size_t offset) {
            packed[offset] = report.success ? 1.0 : 0.0;
            packed[offset + 1] = report.success
                    ? static_cast<double>(report.support.num_inliers)
                    : 0.0;
            packed[offset + 2] = static_cast<double>(report.num_trials);
            packed[offset + 3] =
                    static_cast<double>(report.local_optimization_trials);
            packed[offset + 4] =
                    static_cast<double>(report.accepted_local_refits);
            packed[offset + 5] = report.best_model_is_local ? 1.0 : 0.0;
        };
        pack_report(geometry.epipolar.essential, 10);
        pack_report(geometry.epipolar.fundamental, 16);
        pack_report(geometry.homography, 22);
        packed[28] = geometry.homography.success
                ? geometry.homography.support.residual_sum
                : 0.0;
        if (pose_recovered && pose.success) {
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    packed[29 + static_cast<std::size_t>(row * 3 + col)] =
                            pose.rotation(row, col);
                }
            }
            for (int i = 0; i < 3; ++i) {
                packed[38 + static_cast<std::size_t>(i)] = pose.translation(i);
            }
        }
        const auto pack_mask = [&packed, count](
                const std::vector<char>& mask,
                const std::size_t offset) {
            if (mask.size() != count) {
                return;
            }
            for (std::size_t i = 0; i < count; ++i) {
                packed[offset + i] = mask[i] != 0 ? 1.0 : 0.0;
            }
        };
        if (geometry.epipolar.essential.success) {
            pack_mask(geometry.epipolar.essential.inlier_mask,
                      kInitialPairHeaderSize);
        }
        if (geometry.epipolar.fundamental.success) {
            pack_mask(geometry.epipolar.fundamental.inlier_mask,
                      kInitialPairHeaderSize + count);
        }
        if (geometry.homography.success) {
            pack_mask(geometry.homography.inlier_mask,
                      kInitialPairHeaderSize + 2 * count);
        }
        pack_mask(geometry.decision.selected_inlier_mask,
                  kInitialPairHeaderSize + 3 * count);
        if (pose_recovered && pose.success) {
            pack_mask(pose.cheirality_mask,
                      kInitialPairHeaderSize + 4 * count);
        }
        return MakeDoubleArray(env, packed);
    } catch (const std::exception&) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}
