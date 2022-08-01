#pragma once

#include <psdr/core/bitmap.h>
#include "bsdf.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(Diffuse, final, BSDF)
public:
    Diffuse() : m_reflectance(0.5f) {}
    Diffuse(const ScalarVector3f &ref) : m_reflectance(ref) {}
    Diffuse(const char *refl_file);
    Diffuse(const Bitmap3fD &reflectance);

     void setDiffuseReflectance(std::string filename){
        m_reflectance.load_openexr(filename.c_str());
    }

    SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    // BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const override;
    // BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const override;

    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const override;
    bool anisotropic() const override { return false; }
    bool hasbssdf() const override { return false; }

    std::string to_string() const override { return std::string("Diffuse[id=") + m_id + "]"; }
    void setSigmaS(ScalarVector3f &a){return;};
    void setSigmaA(ScalarVector3f &a){return;};
    Bitmap3fD m_reflectance;

    // Xi Deng added to support bssrdf
    SpectrumC eval(const IntersectionC &its, const BSDFSampleC &bs, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const BSDFSampleD &bs, MaskD active = true) const override;

    BSDFSampleC sample(const Scene *scene, const IntersectionC &its, const Vector8fC &sample, MaskC active = true) const override;
    BSDFSampleD sample(const Scene *scene, const IntersectionD &its, const Vector8fD &sample, MaskD active = true) const override;

    FloatC pdf(const IntersectionC &its, const BSDFSampleC &wo, MaskC active) const override;
    FloatD pdf(const IntersectionD &its, const BSDFSampleD &wo, MaskD active) const override;

    FloatC pdfpoint(const IntersectionC &its, const BSDFSampleC &bs, MaskC active) const override {return 1.0f;}
    FloatD pdfpoint(const IntersectionD &its, const BSDFSampleD &bs, MaskD active) const override {return 1.0f;}

protected:
    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad>&, const Vector3f<ad>&, Mask<ad>) const;

    template <bool ad>
    BSDFSample<ad> __sample(const Intersection<ad>&, const Vector8f<ad>&, Mask<ad>) const;

    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &, const Vector3f<ad> &, Mask<ad>) const;
PSDR_CLASS_DECL_END(Diffuse)
} // namespace psdr
