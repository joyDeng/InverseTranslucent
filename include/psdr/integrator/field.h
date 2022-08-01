#pragma once

#include "integrator.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(FieldExtractionIntegrator, final, Integrator)
public:
    FieldExtractionIntegrator(const char *field);

    std::string m_field;

protected:
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true, int sensor_id = 0) const override;
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true, int sensor_id = 0) const override;

    template <bool ad>
    Spectrum<ad> __Li(const Scene &scene, const Ray<ad> &ray, Mask<ad> active) const;
PSDR_CLASS_DECL_END(FieldExtractionIntegrator)

} // namespace psdr
