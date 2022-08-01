#include <misc/Exception.h>
#include <psdr/core/frame.h>
#include <psdr/core/intersection.h>
#include <psdr/shape/mesh.h>
#include <psdr/emitter/point.h>

namespace psdr
{

void PointLight::configure() {

}

void PointLight::setposition(Vector3fD p){m_position = p;}

SpectrumC PointLight::eval(const IntersectionC &its, MaskC active) const {
    FloatC distance = norm(its.p - detach(m_position));
    return detach(m_power) / (4.0f * Pi * distance * distance);
}


SpectrumD PointLight::eval(const IntersectionD &its, MaskD active) const {
    FloatD distance = norm(its.p - m_position);
    return m_power / (4.0f * Pi * distance * distance);
}


PositionSampleC PointLight::sample_position(const Vector3fC &ref_p, const Vector2fC &sample2, MaskC active) const {
    return __sample_position<false>(ref_p, sample2, active);
}


PositionSampleD PointLight::sample_position(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active) const {
    return __sample_position<true>(ref_p, sample2, active);
}


template <bool ad>
PositionSample<ad> PointLight::__sample_position(const Vector3f<ad> &ref_p, const Vector2f<ad>&sample2, Mask<ad> active) const {
    PositionSample<ad> result;
    if constexpr (ad){
        result.p = Vector3fD(m_position);
        result.n = Vector3fD(normalize(ref_p - result.p));
    } else{
        result.p = Vector3fC(detach(m_position));
        result.n = Vector3fC(normalize(ref_p - result.p));
    }
    result.J = 1.0f;
    result.pdf = 1.0f;
    return result;
}


FloatC PointLight::sample_position_pdf(const Vector3fC &ref_p, const IntersectionC &its, MaskC active) const {
    return FloatC(1.0f);
}


FloatD PointLight::sample_position_pdf(const Vector3fD &ref_p, const IntersectionD &its, MaskD active) const {
    return FloatD(1.0f);
}


template <bool ad>
Float<ad> PointLight::__sample_position_pdf(const Vector3f<ad> &ref_p, const Intersection<ad> &its, Mask<ad> active) const {
    return Float<ad>(1.0f);
}


std::string PointLight::to_string() const {
    std::ostringstream oss;
    oss << "PointLight[power = " << m_power;
    return oss.str();
}

} // namespace psdr
