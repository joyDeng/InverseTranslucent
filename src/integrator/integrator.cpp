#include <chrono>
#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/sensor.h>
#include <psdr/integrator/integrator.h>

namespace psdr
{

SpectrumC Integrator::renderC(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    SpectrumC result = __render<false>(scene, sensor_id);

    cuda_eval(); cuda_sync();

    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        LOG(oss.str().c_str());
    }

    return result;
}


SpectrumD Integrator::renderD(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    // Interior integral

    SpectrumD result = __render<true>(scene, sensor_id);

    // Boundary integral
    if ( likely(scene.m_opts.sppe > 0) ) {
        render_primary_edges(scene, sensor_id, result);
    }

    if ( likely(scene.m_opts.sppse > 0) ) {
        // std::cout<<" secondary edge "<<std::endl;
        render_secondary_edges(scene, sensor_id, result);
    }
    
    cuda_eval(); cuda_sync();

    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        LOG(oss.str().c_str());
    }
    // std::cout<<"renderD: test3"<<std::endl;
    return result;
}


template <bool ad>
Spectrum<ad> Integrator::__render(const Scene &scene, int sensor_id) const {
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.cropwidth*opts.cropheight;
    // std::cout<<"pixel number: "<<num_pixels<<std::endl;
    Spectrum<ad> result = zero<Spectrum<ad>>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());
        // std::cout<<"sample number values: "<<num_samples<<std::endl;
        Int<ad> idx = arange<Int<ad>>(num_samples);
        if ( likely(opts.spp > 1) ) idx /= opts.spp;
        // std::cout<<"idx: "<<slices(idx)<<std::endl;
        Vector2f<ad> samples_base = gather<Vector2f<ad>>(meshgrid(arange<Float<ad>>(opts.cropwidth),
                                                        arange<Float<ad>>(opts.cropheight)),
                                                        idx);

        Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                /ScalarVector2f(opts.cropwidth, opts.cropheight);
        // std::cout<<"sampled: "<<slices(samples)<<std::endl;
        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        // std::cout<<"camera_ray: "<<slices(camera_ray)<<std::endl;
        Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray, true, sensor_id);
        masked(value, ~enoki::isfinite<Spectrum<ad>>(value)) = 0.f;
        // std::cout<<"Li: "<<slices(value)<<std::endl;
        scatter_add(result, value, idx);
        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
        }
    }

    return result;
}


void Integrator::render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
    const RenderOption &opts = scene.m_opts;

    const Sensor *sensor = scene.m_sensors[sensor_id];
    if ( sensor->m_enable_edges ) {
        PrimaryEdgeSample edge_samples = sensor->sample_primary_edge(scene.m_samplers[1].next_1d<false>());
        MaskC valid = (edge_samples.idx >= 0);
#ifdef PSDR_PRIMARY_EDGE_VIS_CHECK
        IntersectionC its = scene.ray_intersect<false>(edge_samples.ray_c);
        valid &= ~its.is_valid();
#endif
        SpectrumC delta_L = Li(scene, scene.m_samplers[1], edge_samples.ray_n, valid, sensor_id) - 
                            Li(scene, scene.m_samplers[1], edge_samples.ray_p, valid, sensor_id);
                            
                            // estimate subgradient
        SpectrumD value = edge_samples.x_dot_n*SpectrumD(delta_L/edge_samples.pdf);
        masked(value, ~enoki::isfinite<SpectrumD>(value)) = 0.f;
        if ( likely(opts.sppe > 1) ) {
            value /= static_cast<float>(opts.sppe);
        }
        value -= detach(value);
        scatter_add(result, value, IntD(edge_samples.idx), valid);
    }
}

} // namespace psdr
