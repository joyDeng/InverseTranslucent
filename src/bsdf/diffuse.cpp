#include <psdr/core/bitmap.h>
#include <psdr/core/warp.h>
#include <psdr/bsdf/diffuse.h>

namespace psdr
{

Diffuse::Diffuse(const char *refl_file) : m_reflectance(refl_file) {}


Diffuse::Diffuse(const Bitmap3fD &reflectance) : m_reflectance(reflectance) {}


SpectrumC Diffuse::eval(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    return __eval<false>(its, wo, active);
}


SpectrumD Diffuse::eval(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    return __eval<true>(its, wo, active);
}


// Begin:  Xi Deng added to support bssrdf
SpectrumC Diffuse::eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const {
    // Vector3fC wo = its.sh_frame.to_world(bs.wo);
    return __eval<false>(its, bs.wo, active);
}


SpectrumD Diffuse::eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const {
    // Vector3fD wo = its.sh_frame.to_world(bs.wo);
    return __eval<true>(its, bs.wo, active);
}

FloatC Diffuse::pdf(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const {
    return __pdf<false>(its, bs.wo, active);
}


FloatD Diffuse::pdf(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const {
    return __pdf<true>(its, bs.wo, active);
}

//


template <bool ad>
Spectrum<ad> Diffuse::__eval(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(wo);

    active &= (cos_theta_i > 0.f && cos_theta_o > 0.f);
    // std::cout<<"cos_theta_i = "<<cos_theta_i<<std::endl;
    // std::cout<<"cos_theta_o = "<<cos_theta_o<<std::endl;
    // std::cout<<woz()<<std::endl;

    Spectrum<ad> value = m_reflectance.eval<ad>(its.uv) * InvPi * cos_theta_o;
    return value & active;
}


BSDFSampleC Diffuse::sample(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active) const {
    return __sample<false>(its, sample, active);
}


BSDFSampleD Diffuse::sample(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active) const {
    return __sample<true>(its, sample, active);;
}


template <bool ad>
BSDFSample<ad> Diffuse::__sample(const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    BSDFSample<ad> bs;
    bs.wo = warp::square_to_cosine_hemisphere<ad>(tail<2>(sample));
    bs.pdf = warp::square_to_cosine_hemisphere_pdf<ad>(bs.wo);
    bs.po = its;
    bs.is_valid = active && (cos_theta_i > 0.f);
    bs.is_sub = false;
    return bs;
}


FloatC Diffuse::pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    return __pdf<false>(its, wo, active);
}


FloatD Diffuse::pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    return __pdf<true>(its, wo, active);
}


template <bool ad>
Float<ad> Diffuse::__pdf(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active) const {
    FloatC cos_theta_i, cos_theta_o;
    if constexpr ( ad ) {
        cos_theta_i = FrameC::cos_theta(detach(its.wi));
        cos_theta_o = FrameC::cos_theta(detach(wo));
    } else {
        cos_theta_i = FrameC::cos_theta(its.wi);
        cos_theta_o = FrameC::cos_theta(wo);
    }
    active &= (cos_theta_i > 0.f && cos_theta_o > 0.f);

    Float<ad> value = InvPi * cos_theta_o;
    return value & active;
}

} // namespace psdr
