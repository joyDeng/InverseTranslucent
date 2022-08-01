#pragma once

#include <psdr/core/bitmap.h>
#include "bsdf.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(LayerSub, final, BSDF)
public:
    LayerSub() : 
    m_alpha_u(0.1f), m_alpha_v(0.1f), m_eta(0.8f), 
    m_albedo(1.0f), m_sigma_t(0.5f), m_reflectance(0.5f), 
    m_specular_reflectance(1.0f), m_g(0.0f), m_layers(3)
        {   
            m_anisotropic = false;
         }

    LayerSub(const Bitmap1fD &alpha, const Bitmap1fD &eta, const Bitmap3fD &reflectance, const Bitmap3fD &sigma_t, const Bitmap3fD &albedo)
        : 
    m_alpha_u(alpha), m_alpha_v(alpha), m_eta(eta),
    m_reflectance(reflectance), m_specular_reflectance(1.0f), 
    m_albedo(albedo), m_sigma_t(sigma_t), m_g(0.0f), m_layers(3){ m_anisotropic = false; }

    void setAlbedoTexture(std::string filename){
        m_albedo.load_openexr(filename.c_str());
    }

    void setSigmaTexture(std::string filename){
        m_sigma_t.load_openexr(filename.c_str());
    }

    void setAlphaTexture(std::string filename){
        m_alpha_u.load_openexr(filename.c_str());
        m_alpha_v.load_openexr(filename.c_str());
    }

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

    std::string to_string() const override { return std::string("LayerSub[id=") + m_id + "]"; }

    Bitmap1fD m_alpha_u, m_alpha_v; // surface roughness
    Bitmap1fD m_specular_prob; // probability of sample specular pdf
    Bitmap1fD m_eta, m_g;
    Bitmap3fD m_albedo, m_sigma_t; // medium features
    Bitmap3fD m_reflectance, m_specular_reflectance; // reflectance
    int m_layers = 3;
    float m_maxDepth = 1.0;
    float m_layerDepth = 0.1;
    
    mutable const Scene * m_scene_pointer = nullptr;
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
    Spectrum<ad> __Sp(const Intersection<ad>&its, const Intersection<ad>&bsp, const Mask<ad> active) const;

    template <bool ad>
    Spectrum<ad> __Sp_inner(const Spectrum<ad> sa, const Spectrum<ad> mi, const Spectrum<ad> eta, const Float<ad> r) const;

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
    // Intersection<ad> __get_intersection(const Scene *scene, const Intersection<ad> &its, const Vector8f<ad> &sample, Vector3f<ad> vx, Vector3f<ad> vy, Vector3f<ad> vz, Float<ad> l, Float<ad> r, Float<ad> phi, Mask<ad> active) const;
    // template <bool ad>
    // Intersection<ad> __selectRandomSample(const Int<ad> randomsample, std::vector<Intersection<ad>> inters) const;
    
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
    template <bool ad>
    Vector3f<ad> refract(const Vector3f<ad> wi, const Frame<ad> frame, const Spectrum<ad> eta) const;
    
    // help functions
    template <bool ad> Spectrum<ad> __FresnelMoment1(Spectrum<ad> eta) const;
    template <bool ad> Spectrum<ad> __FresnelMoment2(Spectrum<ad> eta) const;
    template <bool ad> Spectrum<ad> __FersnelDi(Spectrum<ad> eta_i, Spectrum<ad> eta_o, Float<ad> cos_theta_i) const;

PSDR_CLASS_DECL_END(LayerSub)

} // namespace psdr
