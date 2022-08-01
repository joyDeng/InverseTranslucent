#include <psdr/core/bitmap.h>
#include <psdr/core/warp.h>

namespace psdr {
    SpectrumC BSSDF::eval(const IntersectionC &its, const Vector3fC &wo, const Vector3fC &po, MaskC active) const {
        return SpectrumC(1.0f);//__S<false>(its, wo, po);
    }

    SpectrumD BSSDF::eval(const IntersectionD &its, const Vector3fD &wo, const Vector3fD &po, MaskD active) const {
        return SpectrumD(1.0f);//__S<true>(its, wo, po);
    }

    BSSDFSampleC BSSDF::sample(const IntersectionC &its, const Vector3fC &sampleP, MaskC active) const {
        return __samplePos<false>(its, sampleP, active);
    }

    BSSDFSampleD BSSDF::sample(const IntersectionD &its, const Vector3fD &sampleP, MaskD active) const {
        return __samplePos<true>(its, sampleP, active);
    }

    FloatC BSSDF::pdfPo(const IntersectionC &its, const Vector3fC &po, MaskC active) const {
        return 1.0;
    }

    FloatD BSSDF::pdfPo(const IntersectionD &its, const Vector3fD &po, MaskD active) const {
        return 1.0;
    }

    template <bool ad>
    BSSDFSample<ad> BSSDF::__samplePos(const Intersection<ad> &its, const Vector3f<ad> &sample, Mask<ad> active) const {
        // Choose projection axis for BSSRDF sampling
        // Choose Spectral channel for BSSRDF sampling
        // Sample BSSRDF profile in polar coordinates
        // Compute BSSRDF profile bounds and intersection height
        // Compute BSSRDF sampling ray segment
        // Intersect BSSRDF sampling ray against the scene geometry
        // Randomly choose one of several intersections during BSSRDF sampling
        // Compute sample PDF and return the spatial BSSRDF term Sp
        Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
        BSSDFSample<ad> bs;

        bs.po = warp::square_to_cosine_hemisphere<ad>(tail<2>(sample));
        bs.pdf = warp::square_to_cosine_hemisphere_pdf<ad>(bs.po);
        bs.is_valid = active && (cos_theta_i > 0.f);
        return bs;
    }
}