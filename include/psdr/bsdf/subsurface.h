#pragma once

#include <psdr/core/bitmap.h>
#include "bsdf.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(Subsurface, final, BSDF)
public:
    Subsurface() : 
    m_alpha_u(0.1f), m_alpha_v(0.1f), m_eta(0.8f), 
    m_albedo(1.0f), m_sigma_t(0.5f), m_reflectance(0.5f), 
    m_specular_reflectance(1.0f), m_g(0.0f) 
        {   
            m_anisotropic = false;
         }

    Subsurface(const Bitmap1fD &alpha, const Bitmap1fD &eta, const Bitmap3fD &reflectance, const Bitmap3fD &sigma_t, const Bitmap3fD &albedo)
        : 
    m_alpha_u(alpha), m_alpha_v(alpha), m_eta(eta),
    m_reflectance(reflectance), m_specular_reflectance(1.0f), 
    m_albedo(albedo), m_sigma_t(sigma_t), m_g(0.0f) { m_anisotropic = false; }

    void setAlbedo(ScalarVector3f &albedo){
        m_albedo.fill(albedo);
    }

    void setSigmaT(ScalarVector3f &sigma_t){
        m_sigma_t.fill(sigma_t);
    }

    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    FloatC pdfpoint(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const override;
    FloatD pdfpoint(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const override;

    SpectrumC eval(const IntersectionC &its, const Vector3fC &bs, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &bs, MaskD active = true) const override;

    SpectrumC eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active = true) const override;

    BSDFSampleC sample(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active = true) const override;
    BSDFSampleD sample(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active = true) const override;

    FloatC pdf(const IntersectionC &its, const BSDFSampleC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const BSDFSampleD &wo, MaskD active = true) const override;

    bool anisotropic() const override { return m_anisotropic; }
    bool hasbssdf() const override { return true; }

    std::string to_string() const override { return std::string("Subsurface[id=") + m_id + "]"; }

    Bitmap1fD m_alpha_u, m_alpha_v; // surface roughness
    Bitmap1fD m_specular_prob; // probability of sample specular pdf
    Bitmap1fD m_eta, m_g;
    Bitmap3fD m_albedo, m_sigma_t; // medium features
    Bitmap3fD m_reflectance, m_specular_reflectance; // reflectance
    

    bool m_anisotropic;

protected:
    template <bool ad>
    BSDFSample<ad> __sample(const Scene *scene, const Intersection<ad>&, const Vector8f<ad>&, Mask<ad>) const;
   
    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;
    // template <bool ad>
    // Float<ad> __pdf_better(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;
    template <bool ad>
    Float<ad> __pdfspecular(const Intersection<ad> &, const BSDFSample<ad> &, Mask<ad>) const;

    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad>&, const BSDFSample<ad>&, Mask<ad>) const;

    // Xi Deng added
    template <bool ad>
    Spectrum<ad> __Sp(const Intersection<ad>&its, const BSDFSample<ad>&bs) const;

    // template <bool ad>
    // Spectrum<ad> __Spbetter(const Intersection<ad>&, const BSDFSample<ad>&) const;

    template <bool ad>
    Spectrum<ad> __Sw(const Intersection<ad>&, const Vector3f<ad>&, Mask<ad>) const;

    template <bool ad>
    Float<ad> __pdf_wo(const Intersection<ad> &, const Vector3f<ad> &, Mask<ad>) const;

    template <bool ad>
    Float<ad> __pdf_sr(const Float<ad> &, const Float<ad> &, Mask<ad>) const;

    template <bool ad>
    Float<ad> __sample_sr(const Float<ad> &, const Float<ad> &) const;

    template <bool ad>
    Intersection<ad> __sample_sp(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad>active) const;
    // template <bool ad>
    // Intersection<ad> __sample_sp_better(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Float<ad> &pdf, Mask<ad>active) const;

    template <bool ad>
    Float<ad> __pdf_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    template <bool ad>
    Float<ad> __pdf_bsdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;

    template <bool ad>
    BSDFSample<ad> __sample_bsdf(const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const;
    template <bool ad>
    BSDFSample<ad> __sample_sub(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Mask<ad> active) const;

    template <bool ad>
    Spectrum<ad> __eval_bsdf(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    template <bool ad>
    Spectrum<ad> __eval_sub(const Intersection<ad> &its, const BSDFSample<ad> &bs, Mask<ad> active) const;
    // help functions
    template <bool ad> Spectrum<ad> __FresnelMoment1(Spectrum<ad> eta) const;
    template <bool ad> Spectrum<ad> __FresnelMoment2(Spectrum<ad> eta) const;
    template <bool ad> Spectrum<ad> __FersnelDi(Spectrum<ad> eta_i, Spectrum<ad> eta_o, Float<ad> cos_theta_i) const;

    // Inner test
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

PSDR_CLASS_DECL_END(Subsurface)

} // namespace psdr
