package com.trackstudio.app.tracking

/**
 * COLMAP two-view model configuration selected from robust E, F, and H support.
 *
 * The decision order and strict ratio comparisons follow the pinned COLMAP
 * `EstimateCalibratedTwoViewGeometry` implementation. Android applies its own
 * stricter seed-quality gates after this source-derived model classification.
 */
internal object ColmapTwoViewModelArbitrator {

    fun arbitrate(
        essentialSucceeded: Boolean,
        essentialInliers: Int,
        fundamentalSucceeded: Boolean,
        fundamentalInliers: Int,
        homographySucceeded: Boolean,
        homographyInliers: Int,
        minimumInliers: Int = DEFAULT_MINIMUM_INLIERS,
        minimumEssentialToFundamentalRatio: Double = DEFAULT_MINIMUM_E_F_RATIO,
        maximumHomographyRatio: Double = DEFAULT_MAXIMUM_H_RATIO
    ): Decision {
        require(essentialInliers >= 0 && fundamentalInliers >= 0 && homographyInliers >= 0)
        require(minimumInliers > 0)
        require(minimumEssentialToFundamentalRatio.isFinite() &&
            minimumEssentialToFundamentalRatio in 0.0..1.0)
        require(maximumHomographyRatio.isFinite() && maximumHomographyRatio in 0.0..1.0)

        val essentialToFundamentalRatio = ratio(essentialInliers, fundamentalInliers)
        val homographyToFundamentalRatio = ratio(homographyInliers, fundamentalInliers)
        val homographyToEssentialRatio = ratio(homographyInliers, essentialInliers)
        val anyModelSucceeded = essentialSucceeded || fundamentalSucceeded || homographySucceeded
        val anyModelHasSupport = essentialInliers >= minimumInliers ||
            fundamentalInliers >= minimumInliers || homographyInliers >= minimumInliers

        if (!anyModelSucceeded || !anyModelHasSupport) {
            return Decision(
                Configuration.DEGENERATE, InlierModel.NONE, 0,
                essentialToFundamentalRatio,
                homographyToFundamentalRatio,
                homographyToEssentialRatio
            )
        }

        if (essentialSucceeded &&
            essentialToFundamentalRatio > minimumEssentialToFundamentalRatio &&
            essentialInliers >= minimumInliers
        ) {
            val bestEpipolarModel = if (essentialInliers >= fundamentalInliers) {
                InlierModel.ESSENTIAL to essentialInliers
            } else {
                InlierModel.FUNDAMENTAL to fundamentalInliers
            }
            return if (homographyToEssentialRatio > maximumHomographyRatio) {
                if (homographyInliers > bestEpipolarModel.second) {
                    Decision(
                        Configuration.PLANAR_OR_PANORAMIC,
                        InlierModel.HOMOGRAPHY,
                        homographyInliers,
                        essentialToFundamentalRatio,
                        homographyToFundamentalRatio,
                        homographyToEssentialRatio
                    )
                } else {
                    Decision(
                        Configuration.PLANAR_OR_PANORAMIC,
                        bestEpipolarModel.first,
                        bestEpipolarModel.second,
                        essentialToFundamentalRatio,
                        homographyToFundamentalRatio,
                        homographyToEssentialRatio
                    )
                }
            } else {
                Decision(
                    Configuration.CALIBRATED,
                    bestEpipolarModel.first,
                    bestEpipolarModel.second,
                    essentialToFundamentalRatio,
                    homographyToFundamentalRatio,
                    homographyToEssentialRatio
                )
            }
        }

        if (fundamentalSucceeded && fundamentalInliers >= minimumInliers) {
            return if (homographyToFundamentalRatio > maximumHomographyRatio) {
                if (homographyInliers > fundamentalInliers) {
                    Decision(
                        Configuration.PLANAR_OR_PANORAMIC,
                        InlierModel.HOMOGRAPHY,
                        homographyInliers,
                        essentialToFundamentalRatio,
                        homographyToFundamentalRatio,
                        homographyToEssentialRatio
                    )
                } else {
                    Decision(
                        Configuration.PLANAR_OR_PANORAMIC,
                        InlierModel.FUNDAMENTAL,
                        fundamentalInliers,
                        essentialToFundamentalRatio,
                        homographyToFundamentalRatio,
                        homographyToEssentialRatio
                    )
                }
            } else {
                Decision(
                    Configuration.UNCALIBRATED,
                    InlierModel.FUNDAMENTAL,
                    fundamentalInliers,
                    essentialToFundamentalRatio,
                    homographyToFundamentalRatio,
                    homographyToEssentialRatio
                )
            }
        }

        if (homographySucceeded && homographyInliers >= minimumInliers) {
            return Decision(
                Configuration.PLANAR_OR_PANORAMIC,
                InlierModel.HOMOGRAPHY,
                homographyInliers,
                essentialToFundamentalRatio,
                homographyToFundamentalRatio,
                homographyToEssentialRatio
            )
        }

        return Decision(
            Configuration.DEGENERATE,
            InlierModel.NONE,
            0,
            essentialToFundamentalRatio,
            homographyToFundamentalRatio,
            homographyToEssentialRatio
        )
    }

    private fun ratio(numerator: Int, denominator: Int): Double = when {
        denominator > 0 -> numerator.toDouble() / denominator
        numerator > 0 -> Double.POSITIVE_INFINITY
        else -> Double.NaN
    }

    const val DEFAULT_MINIMUM_INLIERS = 15
    const val DEFAULT_MINIMUM_E_F_RATIO = 0.95
    const val DEFAULT_MAXIMUM_H_RATIO = 0.8

    enum class Configuration {
        CALIBRATED,
        UNCALIBRATED,
        PLANAR_OR_PANORAMIC,
        DEGENERATE
    }

    enum class InlierModel {
        ESSENTIAL,
        FUNDAMENTAL,
        HOMOGRAPHY,
        NONE
    }

    data class Decision(
        val configuration: Configuration,
        val inlierModel: InlierModel,
        val inlierCount: Int,
        val essentialToFundamentalRatio: Double,
        val homographyToFundamentalRatio: Double,
        val homographyToEssentialRatio: Double
    )
}
