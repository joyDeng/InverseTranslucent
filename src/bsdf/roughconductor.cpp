#include <psdr/core/bitmap.h>
#include <psdr/core/warp.h>
#include <psdr/bsdf/ggx.h>
#include <psdr/bsdf/roughconductor.h>

namespace psdr
{

SpectrumC RoughConductor::eval(const IntersectionC& its, const Vector3fC& wo, MaskC active) const {
    return __eval<false>(its, wo, active);
}


SpectrumD RoughConductor::eval(const IntersectionD& its, const Vector3fD& wo, MaskD active) const {
    return __eval<true>(its, wo, active);
}

// Begin: Xi Deng added to support bssrdf
SpectrumC RoughConductor::eval(const IntersectionC& its, const BSDFSampleC& bs, MaskC active) const {
    // Vector3fC wo = its.sh_frame.to_world(bs.wo);
    return __eval<false>(its, bs.wo, active);
}


SpectrumD RoughConductor::eval(const IntersectionD& its, const BSDFSampleD& bs, MaskD active) const {
    // Vector3fD wo = its.sh_frame.to_world(bs.wo);
    return __eval<true>(its, bs.wo, active);
}

FloatC RoughConductor::pdf(const IntersectionC& its, const BSDFSampleC& bs, MaskC active) const {
    return __pdf<false>(its, bs.wo, active);
}


FloatD RoughConductor::pdf(const IntersectionD& its, const BSDFSampleD& bs, MaskD active) const {
    return __pdf<true>(its, bs.wo, active);
}

// End

BSDFSampleC RoughConductor::sample(const Scene *scene, const IntersectionC& its, const Vector8fC& sample, MaskC active) const {
    return __sample<false>(its, sample, active);
}


BSDFSampleD RoughConductor::sample(const Scene *scene, const IntersectionD& its, const Vector8fD& sample, MaskD active) const {
    return __sample<true>(its, sample, active);
}


FloatC RoughConductor::pdf(const IntersectionC& its, const Vector3fC& wo, MaskC active) const {
    return __pdf<false>(its, wo, active);
}


FloatD RoughConductor::pdf(const IntersectionD& its, const Vector3fD& wo, MaskD active) const {
    return __pdf<true>(its, wo, active);
}


template <bool ad>
Spectrum<ad> RoughConductor::__eval(const Intersection<ad>& its, const Vector3f<ad>& wo, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(wo);
    active &= cos_theta_i > 0.f && cos_theta_o > 0.f;
    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_v.eval<ad>(its.uv);
    GGXDistribution m_distr(alpha_u, alpha_v);
    Vector3f<ad> H = normalize(wo + its.wi);
    Float<ad> D = m_distr.eval<ad>(H);
    active &= neq(D, 0.f);
    Float<ad> G = m_distr.G<ad>(its.wi, wo, H);
    Spectrum<ad> result = D * G / (4.f * Frame<ad>::cos_theta(its.wi));
    Spectrum<ad> F = fresnel<ad>(m_eta.eval<ad>(its.uv), m_k.eval<ad>(its.uv), dot(its.wi, H));
    Spectrum<ad> specular_reflectance = m_specular_reflectance.eval<ad>(its.uv);

    return (F * result * specular_reflectance) & active;
}


template <bool ad>
Float<ad> RoughConductor::__pdf(const Intersection<ad>& its, const Vector3f<ad>& wo, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(wo);

    Spectrum<ad> m = normalize(wo + its.wi);
    
    active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
              dot(its.wi, m) > 0.f && dot(wo, m) > 0.f;

    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_v.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);
    Float<ad> result = distr.eval<ad>(m) * distr.smith_g1<ad>(its.wi, m) /
                       (4.f * cos_theta_i);
    return result;
}


template <bool ad>
BSDFSample<ad> RoughConductor::__sample(const Intersection<ad>& its, const Vector8f<ad>& sample, Mask<ad> active) const {
    BSDFSample<ad> bs;
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_v.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);

    Vector3f<ad> wo = distr.sample<ad>(its.wi, tail<3>(sample));
    bs.wo = fmsub(Vector3f<ad>(wo), 2.f * dot(its.wi, wo), its.wi);
    Vector3f<ad> H = normalize(bs.wo + its.wi);
    // bs.pdf = pdf(its, bs.wo, active);
    bs.po = its;
    bs.pdf = pdf(its, bs, active);
    bs.is_valid = (cos_theta_i > 0.f) & active;
    bs.is_sub = false;
    return bs;
}

} // namespace psdr
