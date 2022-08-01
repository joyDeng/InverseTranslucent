#pragma once

#include <psdr/core/bitmap.h>
#include "bsdf.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(RoughConductor, final, BSDF)
public:
    RoughConductor()
        : m_alpha_u(0.1f), m_alpha_v(0.1f), m_eta(0.0f), m_k(1.0f), m_specular_reflectance(1.0f) { m_anisotropic = false; }

    RoughConductor(const Bitmap1fD &alpha, const Bitmap3fD &eta, const Bitmap3fD &k)
        : m_alpha_u(alpha), m_alpha_v(alpha), m_eta(eta), m_k(k), m_specular_reflectance(1.0f) { m_anisotropic = false; }

    RoughConductor(const Bitmap1fD &alpha, const Bitmap3fD &eta, const Bitmap3fD &k, const Bitmap3fD &sr)
        : m_alpha_u(alpha), m_alpha_v(alpha), m_eta(eta), m_k(k), m_specular_reflectance(sr) { m_anisotropic = false; }

    RoughConductor(const Bitmap1fD &alpha_u, const Bitmap1fD &alpha_v, const Bitmap3fD &eta, const Bitmap3fD &k)
        : m_alpha_u(alpha_u), m_alpha_v(alpha_v), m_eta(eta), m_k(k), m_specular_reflectance(1.0f) { m_anisotropic = true; }

    RoughConductor(const Bitmap1fD &alpha_u, const Bitmap1fD &alpha_v, const Bitmap3fD &eta, const Bitmap3fD &k, const Bitmap3fD &sr)
        : m_alpha_u(alpha_u), m_alpha_v(alpha_v), m_eta(eta), m_k(k), m_specular_reflectance(sr) { m_anisotropic = true; }

    SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    // Begin: Xi Deng added to support bssrdf
    SpectrumC eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active = true) const override;
    FloatC pdf(const IntersectionC &its, const BSDFSampleC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const BSDFSampleD &wo, MaskD active = true) const override;
    FloatC pdfpoint(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const override {return 1.0f;}
    FloatD pdfpoint(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const override {return 1.0f;}

    // End
    void setSigmaS(ScalarVector3f &a) {return;}
    void setSigmaA(ScalarVector3f &a) {return;}

    BSDFSampleC sample(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active = true) const override;
    BSDFSampleD sample(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active = true) const override;

    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

   
    bool anisotropic() const override { return m_anisotropic; }
    bool hasbssdf() const override { return false; }

    std::string to_string() const override { return std::string("RoughConductor[id=") + m_id + "]"; }

    Bitmap1fD m_alpha_u, m_alpha_v;
    Bitmap3fD m_eta, m_k;
    Bitmap3fD m_specular_reflectance;
    bool m_anisotropic;

protected:
    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad>&, const Vector3f<ad>&, Mask<ad>) const;

    template <bool ad>
    BSDFSample<ad> __sample(const Intersection<ad>&, const Vector8f<ad>&, Mask<ad>) const;

    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &, const Vector3f<ad> &, Mask<ad>) const;
PSDR_CLASS_DECL_END(RoughConductor)

} // namespace psdr
