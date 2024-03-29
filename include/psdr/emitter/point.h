#pragma once

#include <psdr/psdr.h>
#include "emitter.h"
#include <psdr/core/transform.h>

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(PointLight, final, Emitter)
public:
    PointLight(const ScalarVector3f &power, const Matrix4fD trans) : m_power(power) {
        m_position = transform_pos(trans, zero<Vector3fD>()); 
    }

    void configure() override;
    void setposition(Vector3fD p) override;
    

    SpectrumC eval(const IntersectionC &its, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, MaskD active = true) const override;

    PositionSampleC sample_position(const Vector3fC &ref_p, const Vector2fC &sample2, MaskC active = true) const override;
    PositionSampleD sample_position(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active = true) const override;

    FloatC sample_position_pdf(const Vector3fC &ref_p, const IntersectionC &its, MaskC active = true) const override;
    FloatD sample_position_pdf(const Vector3fD &ref_p, const IntersectionD &its, MaskD active = true) const override;

    std::string to_string() const override;

    SpectrumD m_power;
    Vector3fD m_position;

    ENOKI_PINNED_OPERATOR_NEW(FloatD)

protected:
    template <bool ad>
    PositionSample<ad> __sample_position(const Vector3f<ad> &ref_p, const Vector2f<ad>&, Mask<ad>) const;

    template <bool ad>
    Float<ad> __sample_position_pdf(const Vector3f<ad>&, const Intersection<ad>&, Mask<ad>) const;

PSDR_CLASS_DECL_END(PointLight)

} // namespace psdr
