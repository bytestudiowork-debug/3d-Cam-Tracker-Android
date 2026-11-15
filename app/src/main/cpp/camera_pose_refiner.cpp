#include <jni.h>

#include "camera_pose_refiner_core.h"

#include <cstddef>
#include <exception>
#include <limits>
#include <vector>

namespace {

jdoubleArray EmptyResult(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_CameraPoseRefiner_nativeRefine(
        JNIEnv* env,
        jobject,
        jdoubleArray object_points_array,
        jdoubleArray image_points_array,
        jdoubleArray intrinsics_array,
        jdoubleArray initial_pose_array,
        jdouble huber_scale_px,
        jint max_iterations,
        jdouble gradient_tolerance) {
    try {
        if (!object_points_array || !image_points_array || !intrinsics_array ||
            !initial_pose_array) {
            return EmptyResult(env);
        }

        const jsize object_values = env->GetArrayLength(object_points_array);
        const jsize image_values = env->GetArrayLength(image_points_array);
        if (object_values < 12 || object_values % 3 != 0 ||
            object_values / 3 > std::numeric_limits<jsize>::max() / 2 ||
            image_values != (object_values / 3) * 2 ||
            env->GetArrayLength(intrinsics_array) != 5 ||
            env->GetArrayLength(initial_pose_array) != 6 || max_iterations <= 0) {
            return EmptyResult(env);
        }

        std::vector<double> object_points(static_cast<size_t>(object_values));
        std::vector<double> image_points(static_cast<size_t>(image_values));
        std::vector<double> intrinsics(5);
        std::vector<double> initial_pose(6);
        env->GetDoubleArrayRegion(object_points_array, 0, object_values, object_points.data());
        env->GetDoubleArrayRegion(image_points_array, 0, image_values, image_points.data());
        env->GetDoubleArrayRegion(intrinsics_array, 0, 5, intrinsics.data());
        env->GetDoubleArrayRegion(initial_pose_array, 0, 6, initial_pose.data());
        if (env->ExceptionCheck()) return nullptr;

        tracking::PoseRefinementResult refinement;
        if (!tracking::RunPoseRefinement(object_points, image_points, intrinsics,
                                         initial_pose, huber_scale_px, max_iterations,
                                         gradient_tolerance, &refinement)) {
            return EmptyResult(env);
        }

        const jdouble output[10] = {
                1.0,
                refinement.initial_cost,
                refinement.final_cost,
                static_cast<double>(refinement.iterations),
                refinement.pose[0], refinement.pose[1], refinement.pose[2],
                refinement.pose[3], refinement.pose[4], refinement.pose[5]};
        jdoubleArray result = env->NewDoubleArray(10);
        if (result == nullptr) return nullptr;
        env->SetDoubleArrayRegion(result, 0, 10, output);
        if (env->ExceptionCheck()) return nullptr;
        return result;
    } catch (const std::exception&) {
        return EmptyResult(env);
    } catch (...) {
        return EmptyResult(env);
    }
}
