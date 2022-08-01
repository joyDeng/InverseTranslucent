#pragma once

#include <psdr/core/bitmap.h>
#include "bsdf.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(Microfacet, final, BSDF)
public:
    Microfacet() : m_specularReflectance(0.5f), m_diffuseReflectance(0.5f), m_roughness(0.5f) {}

    Microfacet(const ScalarVector3f &specularRef, const ScalarVector3f &diffuseRef, float roughnessRef) :
        m_specularReflectance(specularRef), m_diffuseReflectance(diffuseRef), m_roughness(roughnessRef) {}

    Microfacet(const char *spec_refl_file, const char *diff_refl_file, const char *roughness_file);

    Microfacet(const Bitmap3fD &spec_refl, const Bitmap3fD &diff_refl, const Bitmap1fD &roughness);

    void setDiffuseReflectance(std::string filename){
        m_diffuseReflectance.load_openexr(filename.c_str());
    }

    void setSpecularReflectance(std::string filename){
        m_specularReflectance.load_openexr(filename.c_str());
    }

    void setAlphaTexture(std::string filename){
        m_roughness.load_openexr(filename.c_str());
    }


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

    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const override;

    bool anisotropic() const override { return false; }
    bool hasbssdf() const override { return false; }

    std::string to_string() const override { return std::string("Microfacet[id=") + m_id + "]"; }

    Bitmap3fD m_specularReflectance,
              m_diffuseReflectance;
    Bitmap1fD m_roughness;

protected:
    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active = true) const;

    template <bool ad>
    BSDFSample<ad> __sample(const Intersection<ad>&, const Vector8f<ad>&, Mask<ad>) const;

    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &, const Vector3f<ad> &, Mask<ad>) const;
PSDR_CLASS_DECL_END(Microfacet)

} // namespace psdr
