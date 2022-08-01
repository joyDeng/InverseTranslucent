#pragma once

#include <psdr/psdr.h>
#include <psdr/core/intersection.h>
#include <psdr/core/records.h>

namespace psdr
{

template <typename Float_>
struct BSSDFSample_ : public SampleRecord_<Float_> {
    PSDR_IMPORT_BASE(SampleRecord_<Float_>, ad, pdf, is_valid)

    Vector3f<ad> po;

    ENOKI_DERIVED_STRUCT(BSSDFSample_, Base,
        ENOKI_BASE_FIELDS(pdf, is_valid),
        ENOKI_DERIVED_FIELDS(po)
    )
};


PSDR_CLASS_DECL_BEGIN(BSSDF,,Object)
public:
    
    ~BSSDF() override {}
    BSSDF() : m_eta(1.0f) {};
    BSSDF( const FloatD eta ) : m_eta(eta) {};

    std::string to_string() const override {
        return std::string("BSSDF");
    };

    SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, const Vector3fC &po, MaskC active = true) const;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, const Vector3fD &po, MaskD active = true) const;

    BSSDFSampleC sample(const IntersectionC &its, const Vector3fC &sampleP, MaskC active = true) const;
    BSSDFSampleD sample(const IntersectionD &its, const Vector3fD &sampleP, MaskD active = true) const;

    // FloatC pdf(const IntersectionC &its, const Vector3fC &wo, const Vector3fC &po, MaskC active) const;
    // FloatD pdf(const IntersectionD &its, const Vector3fD &wo, const Vector3fD &po, MaskD active) const;

    FloatC pdfPo(const IntersectionC &its, const Vector3fC &po, MaskC active = true) const;
    FloatD pdfPo(const IntersectionD &its, const Vector3fD &po, MaskD active = true) const;

    

    // virtual FloatC pdfWo(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const;
    // virtual FloatD pdfWo(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const;
    template <bool ad>
    BSSDFSample<ad> __samplePos(const Intersection<ad> &its, const Vector3f<ad> &pos, Mask<ad> active) const;

    // template <bool ad>
    // Float<ad> __FresnelMoment1(Float<ad> eta) const { 
    //     Float<ad> eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    //     if constexpr (eta < 1.0f)
    //         return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
    //            2.49277f * eta4 - 0.68441f * eta5;
    //     else
    //         return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
    //            1.27198f * eta4 + 0.12746f * eta5;
    //  };

    // template <bool ad>
    // Float<ad> __FresnelMoment2(Float<ad> eta) const { 
    //     Float<ad> eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    //     if constexpr (eta < 1.0f)
    //          return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 +
    //            0.07883f * eta4 + 0.04860f * eta5;
    //     else{
    //         Float<ad> r_eta = 1 / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
    //         return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 +
    //            458.843f * r_eta + 404.557f * eta - 189.519f * eta2 +
    //            54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5;
    //     }
    //  };

    // template <bool ad>
    // Spectrum<ad> __S(const Intersection<ad>& its, const Vector3f<ad>&wo, const Vector3f<ad>&po) const {
    //     // TODO check whether the its.n is shframe n
    //     Spectrum<ad> F = 1.0f - fresnel<ad>(zero<Spectrum<ad>>()+m_eta, 1.0f, dot(its.p, its.n));
    //     return F * __Sp<ad>(its, po) * __Sw<ad>(wo);
    // };

    // template <bool ad>
    // Float<ad> __Sp(const Intersection<ad>&its, const Vector3f<ad> &po) const {
    //     Float<ad> distance = norm(po - its.p);
    //     return __Sr<ad>(distance);
    // };

    // template <bool ad>
    // Float<ad> __Sr(Float<ad> distance) const { return 1.0 / distance; };

    // template <bool ad>
    // Spectrum<ad> __Sw(const Vector3f<ad> &w) const{
    //     // TODO replace costheta
    //     // if constexpr (ad) {
    //     Float<ad> eta = zero<Float<ad>>()+detach(m_eta);
    //     Float<ad> c = 1.0 - 2.0 * __FresnelMoment1<ad>(eta); 
    //     return (1.0 - fresnel<ad>( zero<Spectrum<ad>>()+m_eta, 1.0f, Frame<ad>::cos_theta(w))) / (c * Pi);
    //     // } else {
    //     //     FloatC c = 1.0 - 2.0 * __FresnelMoment1<false>(detach(m_eta));            
    //     //     return (1.0 - fresnelFloat<ad>( m_eta, 1.0f, Frame<ad>::cos_theta(w))) / (c * Pi);
    //     // }
    // };
    
    FloatD m_eta;

PSDR_CLASS_DECL_END(BSSDF)
} // namespace psdr

ENOKI_STRUCT_SUPPORT(psdr::BSSDFSample_, pdf, is_valid, po)

ENOKI_CALL_SUPPORT_BEGIN(psdr::BSSDF)
    ENOKI_CALL_SUPPORT_METHOD(eval)
    ENOKI_CALL_SUPPORT_METHOD(sample)
    ENOKI_CALL_SUPPORT_METHOD(pdfPo)
    // ENOKI_CALL_SUPPORT_METHOD(anisotropic)
ENOKI_CALL_SUPPORT_END(psdr::BSSDF)
