#include <misc/Exception.h>
#include <psdr/core/cube_distrb.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/core/transform.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/emitter.h>
#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/perspective.h>
#include <psdr/integrator/direct.h>
#include <psdr/sensor/sensor.h>

namespace psdr
{

template <bool ad>
static inline Float<ad> mis_weight(const Float<ad> &pdf1, const Float<ad> &pdf2) {
    Float<ad> w1 = sqr(pdf1), w2 = sqr(pdf2);
    return w1/(w1 + w2);
}


DirectIntegrator::~DirectIntegrator() {
    for ( auto *item : m_warpper ) {
        if ( item != nullptr ) delete item;
    }
}


DirectIntegrator::DirectIntegrator(int bsdf_samples, int light_samples) : m_bsdf_samples(bsdf_samples), m_light_samples(light_samples) {
    PSDR_ASSERT((bsdf_samples >= 0) && (light_samples >= 0) && (bsdf_samples + light_samples > 0));
}


SpectrumC DirectIntegrator::Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active, int sensor_id) const {
    return __Li<false>(scene, sampler, ray, active);
}


SpectrumD DirectIntegrator::Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active, int sensor_id) const {
    return __Li<true>(scene, sampler, ray, active);
}


template <bool ad>
Spectrum<ad> DirectIntegrator::__Li(const Scene &scene, Sampler &sampler, const Ray<ad> &ray, Mask<ad> active) const {
    std::cout<<"rendering ... "<<std::endl;

    Intersection<ad> its = scene.ray_intersect<ad>(ray, active);

    std::cout<<"intersection ... "<<std::endl;
    active &= its.is_valid();

    Spectrum<ad> result = zero<Spectrum<ad>>();
    BSDFArray<ad> bsdf_array = its.shape->bsdf(active);

    Mask<ad> maskl = Mask<ad>(active);
    Vector3f<ad> lightpoint = (scene.m_emitters[0]->sample_position(Vector3f<ad>(1.0f), Vector2f<ad>(1.0f), maskl)).p;

    BSDFSample<ad> bs;
    bs = bsdf_array->sample(&scene, its, sampler.next_nd<8, ad>(), active);
    Mask<ad> active1 = active && bs.is_valid;

    Vector3f<ad> wo = lightpoint - bs.po.p;
    Float<ad> dist_sqr = squared_norm(wo);
    Float<ad> dist = safe_sqrt(dist_sqr);
    wo /= dist;

    bs.wo = bs.po.sh_frame.to_local(wo);
    bs.po.wi = bs.wo;
    Ray<ad> ray1(bs.po.p, wo, dist);
    Intersection<ad> its1 = scene.ray_intersect<ad, ad>(ray1, active1);
    active1 &= !its1.is_valid();
    Spectrum<ad> bsdf_val;
    if constexpr ( ad ) {
        bsdf_val = bsdf_array->eval(its, bs, active1);
    } else {
        bsdf_val = bsdf_array->eval(its, bs, active1);
    }
    
    Float<ad> pdfpoint = bsdf_array->pdfpoint(its, bs, active1);
    Spectrum<ad> Le = scene.m_emitters[0]->eval(bs.po, active1);

    // std::cout<<" common bsdf_val "<<hsum(hsum(bsdf_val))<<std::endl;

    masked(result, active1) +=  bsdf_val * Le / detach(pdfpoint); //Spectrum<ad>(detach(dd));  //bsdf_val;//detach(Intensity) / 
    return result;
}


void DirectIntegrator::preprocess_secondary_edges(const Scene &scene, int sensor_id, const ScalarVector4i &reso, int nrounds) {
    PSDR_ASSERT(nrounds > 0);
    PSDR_ASSERT_MSG(scene.is_ready(), "Scene needs to be configured!");

    if ( static_cast<int>(m_warpper.size()) != scene.m_num_sensors )
        m_warpper.resize(scene.m_num_sensors, nullptr);

    if ( m_warpper[sensor_id] == nullptr )
        m_warpper[sensor_id] = new HyperCubeDistribution3f();
    auto warpper = m_warpper[sensor_id];

    warpper->set_resolution(head<3>(reso));
    int num_cells = warpper->m_num_cells;
    const int64_t num_samples = static_cast<int64_t>(num_cells)*reso[3];
    PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

    IntC idx = divisor<int>(reso[3])(arange<IntC>(num_samples));
    Vector3iC sample_base = gather<Vector3iC>(warpper->m_cells, idx);

    Sampler sampler;
    sampler.seed(arange<UInt64C>(num_samples));

    FloatC result = zero<FloatC>(num_cells);
    for ( int j = 0; j < nrounds; ++j ) {
        SpectrumC value0;
        std::tie(std::ignore, value0) = eval_secondary_edge<false>(scene, *scene.m_sensors[sensor_id],
                                                                   (sample_base + sampler.next_nd<3, false>())*warpper->m_unit);
        masked(value0, ~enoki::isfinite<SpectrumC>(value0)) = 0.f;
        if ( likely(reso[3] > 1) ) {
            value0 /= static_cast<float>(reso[3]);
        }
        //PSDR_ASSERT(all(hmin(value0) > -Epsilon));
        scatter_add(result, hmax(value0), idx);
    }
    if ( nrounds > 1 ) result /= static_cast<float>(nrounds);
    warpper->set_mass(result);

    cuda_eval(); cuda_sync();
}

void DirectIntegrator::render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
    
    const RenderOption &opts = scene.m_opts;

    Vector3fC sample3 = scene.m_samplers[2].next_nd<3, false>();
    FloatC pdf0 = (m_warpper.empty() || m_warpper[sensor_id] == nullptr) ?
                  1.f : m_warpper[sensor_id]->sample_reuse(sample3);
    // xd added: sample primary ray
    // const int num_pixels = opts.width*opts.height;
    // int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.sppsce;
    // PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());
    // IntC idx = arange<IntC>(num_samples);
    // if ( likely(opts.sppsce > 1) ) idx /= opts.sppsce;

    // Vector2fC samples_base = gather<Vector2fC>(meshgrid(arange<FloatC>(opts.width),
    //                                                     arange<FloatC>(opts.height)),
    //                                                     idx);

    // Vector2fC samples = (samples_base + scene.m_samplers[3].next_2d<false>())
    //                         /ScalarVector2f(opts.width, opts.height);
    // RayC camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
    // IntersectionC its = scene.ray_intersect(camera_ray, true);

    // SpectrumD value = zero<SpectrumD>(num_samples);
    // eval_secondary_edge_bssrdf<true>(scene, its, *scene.m_sensors[sensor_id], sample3, value);
    //end: xd added: sample primary ray

    auto [idx, value] = eval_secondary_edge<true>(scene, *scene.m_sensors[sensor_id], sample3);
    // std::cout<<"values"<<hsum(hsum(value))<<std::endl;
    masked(value, ~enoki::isfinite<SpectrumD>(value)) = 0.f;
    masked(value, pdf0 > Epsilon) /= pdf0;
    if ( likely(opts.sppse > 1) ) {
        value /= (static_cast<float>(opts.sppse) / static_cast<float>(opts.cropwidth * opts.cropheight)); /// 
        // ?
    }

    // if ( likely(opts.sppsce > 1)) {
    //     value /= static_cast<float>(opts.sppsce);
    // }
    
    scatter_add(result, value, IntD(idx), idx >= 0);
}

IntC DirectIntegrator::_compress(IntC input, MaskC mask) const {
    IntC output;
    // output.set_slices(slices(input))
    // int *ptr = slice_ptr(output, 0);
    output = input.compress_(mask);

    // uint64_t count = compress(ptr, input, mask);
    // output.set_slices(count);
    // std::cout<<"slices "<<slices(output)<<" before: "<<slices(input)<<std::endl;
    return output;
    
    // int64_t origin_size = slices(its);
    // int64_t final_size = 0;

    // IntC idx = arange<IntC>(origin_size);
    // its.pixelIdx = idx;

    // IntersectionC compressed;
    // set_slices(compressed, origin_size);
    // Intersection_<float*> *ptr = slice_ptr(compressed, 0);

    // for (int64_t i = 0 ; i < packets(its) ; ++i){
    //     auto input_p = packet(its, i);
    //     final_size += compress(ptr, input_p, input_p.is_valid());
    // }
    // set_slices(compressed, final_size);
    // return compressed;    
}

template <bool ad>
void DirectIntegrator::eval_secondary_edge_bssrdf(const Scene &scene, const IntersectionC &its, const Sensor &sensor, const Vector3fC &sample3, SpectrumD &result) const {
    // sample p0 on edge    
    IntersectionC camera_its = its;
    BoundarySegSampleDirect bss = scene.sample_boundary_segment_direct(sample3);
    MaskC valid = bss.is_valid;

    const Vector3fC &_p0    = detach(bss.p0);
    Vector3fC       _dir    = normalize(bss.p2 - _p0), 
                    _p2     = Vector3fC(0.0f);
    // &_p2    = bss.p2,
    set_slices(_p2, slices(bss.p0));
    _p2 = _p2 + bss.p2;
              
    // check visibility between edge point and light point.
    IntersectionC _its2;
    TriangleInfoD tri_info;
    _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
    valid &= (~_its2.is_valid());

    // trace in another direction to find shadow edge (record the triangle it intersect)
      
    IntersectionC its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), valid, &tri_info);  //scene.ray_intersect<true, false>(RayD(_p0, normalize(_p0 - bss.p2)), MaskD(valid));
    valid &= its1.is_valid();

    // compute idx for matrix entry
    IntC cameraIdx = arange<IntC>(slices(camera_its));
    IntC edgeIdx = arange<IntC>(slices(valid));

    // remove the invalid entry
    if(!any(camera_its.is_valid()) || ! any(valid)) return;
    IntC validCameraSampleIdx = cameraIdx.compress_(camera_its.is_valid());
    IntC validEdgeSampleIdx = edgeIdx.compress_(valid);

    std::cout<<"valid edge sample "<<slices(validEdgeSampleIdx)<<std::endl;
    std::cout<<"valid camera sample "<<slices(validCameraSampleIdx)<<std::endl;


    IntersectionC cameraIts = gather<IntersectionC>(camera_its, validCameraSampleIdx);
    IntersectionC edgeItsC = gather<IntersectionC>(its1, IntC(validEdgeSampleIdx));
    // IntersectionC edgeItsC = detach(edgeItsD);
    TriangleInfoD triangleInfo = gather<TriangleInfoD>(tri_info, IntD(validEdgeSampleIdx));

    Vector3fD p0d = gather<Vector3fD>(bss.p0, IntD(validEdgeSampleIdx));
    Vector3fC p2 = gather<Vector3fC>(_p2, validEdgeSampleIdx);
    std::cout<<"slices p2 "<<slices(p2)<<"slices bss.p2"<<slices(bss.p2)<<"slices p0d"<<slices(p0d)<<std::endl;
    Vector3fC p0 = gather<Vector3fC>(_p0, validEdgeSampleIdx);
    Vector3fC dir = gather<Vector3fC>(_dir, validEdgeSampleIdx);
    Vector3fC bssN = gather<Vector3fC>(bss.n, validEdgeSampleIdx);
    Vector3fC bssE = gather<Vector3fC>(bss.edge, validEdgeSampleIdx);
    Vector3fC bssE2 = gather<Vector3fC>(bss.edge2, validEdgeSampleIdx);
    Vector3fC &p1 = edgeItsC.p;
    FloatC bssPDF = gather<FloatC>(bss.pdf, validEdgeSampleIdx);
    MaskC edgeValid = gather<MaskC>(valid, validEdgeSampleIdx);
    
    
    // compute jacobian that transform from x_b to x_d
    // J = d L(x_d) / d L(x_b) = |x_d - light source| / | x_b - light_source | * sinphi / sinphi2
    FloatC      dist    = norm(p2 - p1),
                dist2   = norm(p0 - p2),
                cos2    = abs(dot(edgeItsC.sh_frame.n, dir));
    Vector3fC   e       = cross(bssE, -dir);
    FloatC      sinphi  = norm(e);
    Vector3fC   proj    = normalize(cross(e, edgeItsC.sh_frame.n));
    FloatC      sinphi2 = norm(cross(-dir, proj));
    FloatC      base_v  =   ( dist / dist2)*(sinphi/sinphi2); //dist2 * dist2 * 
    e = normalize(e);

    edgeValid &= (sinphi > Epsilon) && (sinphi2 > Epsilon);

    // connect to camera all ray
    int64_t numIntersections = slices(cameraIts);
    int64_t numEdgesamples = slices(edgeItsC);
    int64_t num_entry = numIntersections * numEdgesamples;

    IntC cameraSampleMatrixIdx   = arange<IntC>(num_entry);
    IntC sidx   = arange<IntC>(num_entry);

    if (likely(numEdgesamples > 1)){
        cameraSampleMatrixIdx /= numEdgesamples;
        sidx /= numEdgesamples;
        sidx *= numEdgesamples;
    }

    IntC edgeSampleMatrixIdx = arange<IntC>(num_entry);
    edgeSampleMatrixIdx = edgeSampleMatrixIdx - sidx;
    

    IntersectionC   mCameraIts = gather<IntersectionC>(cameraIts, cameraSampleMatrixIdx);
    IntC            rIdx = gather<IntC>(validCameraSampleIdx, cameraSampleMatrixIdx);

    // IntersectionD       mEdgeItsD = gather<IntersectionD>(edgeItsD, IntD(edgeSampleMatrixIdx));
    IntersectionC      mEdgeItsC = gather<IntersectionC>(edgeItsC, edgeSampleMatrixIdx);
    FloatC            mBase_v = gather<FloatC>(base_v, edgeSampleMatrixIdx);
    Vector3fC           projs = gather<Vector3fC>(proj, edgeSampleMatrixIdx);
    Vector3fC              es = gather<Vector3fC>(e, edgeSampleMatrixIdx);
    Vector3fC           _dirs = gather<Vector3fC>(dir, edgeSampleMatrixIdx);
    FloatC              dists = gather<FloatC>(dist, edgeSampleMatrixIdx);
    FloatC               pdfs = gather<FloatC>(bssPDF, edgeSampleMatrixIdx);
    Vector3fD               p0s = gather<Vector3fD>(p0d, IntD(edgeSampleMatrixIdx));
    // std::cout<<" slices p2 "<<slices(p2)<<std::endl;
    Vector3fC               p2s = gather<Vector3fC>(p2, edgeSampleMatrixIdx);
    MaskC        active_entry = gather<MaskC>(edgeValid, edgeSampleMatrixIdx);
    Vector3fC          edge2s = gather<Vector3fC>(bssE2, edgeSampleMatrixIdx);
    Vector3fC              ns = gather<Vector3fC>(edgeItsC.n, edgeSampleMatrixIdx);
    TriangleInfoD triInfos = gather<TriangleInfoD>(triangleInfo, IntD(edgeSampleMatrixIdx));

    // compute bssrdf
    BSDFSampleC bs;
    BSDFArrayC bsdf_array = mEdgeItsC.shape->bsdf(active_entry);
    bs.po = mEdgeItsC;
    bs.wo = mEdgeItsC.wi;
    bs.is_sub = bsdf_array->hasbssdf();
    bs.is_valid = mEdgeItsC.is_valid();
    bs.pdf = FloatC(1.0f);

    active_entry &= mCameraIts.is_valid();
    active_entry &= bs.is_valid;
    SpectrumC bsdf_val = bsdf_array->eval(mCameraIts, bs, active_entry);
    SpectrumC Le = scene.m_emitters[0]->eval(bs.po, active_entry);
    SpectrumC v0 = ( Le * bsdf_val * mBase_v / (pdfs) ) & active_entry;

    if constexpr (ad) {
        Vector3fC nc = normalize(cross(ns, projs));
        v0 *= sign(dot(es, edge2s))*sign(dot(es, nc));
        
        const Vector3fD &v = triInfos.p0,
                        &e1 = triInfos.e1,
                        &e2 = triInfos.e2;

        // std::cout<<"slices p2s"<<slices(p2s)<<std::endl;
        // RayD shadow_ray(Vector3fD(p2s), normalize(-_dirs));
        // Vector2fD material_uv;
        // std::tie(material_uv, std::ignore) = ray_intersect_triangle<true>(v, e1, e2, shadow_ray);
        // Vector3fD point1 = bilinear<true>(detach(v), detach(e1), detach(e2), material_uv);

        RayD shadow_ray2(Vector3fD(p2s), normalize(p0s - p2s));
        Vector2fD material_uv2;
        // FloatD ts;
        std::tie(material_uv2, std::ignore) = ray_intersect_triangle<true>(v, e1, e2, shadow_ray2);
        Vector3fD point2 = bilinear<true>(detach(v), detach(e1), detach(e2), material_uv2);
        
        
        SpectrumD value = (dot(Vector3fD(nc), point2)) * SpectrumD(v0) & active_entry;
        scatter_add(result, value - detach(value), IntD(rIdx));
    }
}


template <bool ad>
std::pair<IntC, Spectrum<ad>> DirectIntegrator::eval_secondary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const {
    BoundarySegSampleDirect bss = scene.sample_boundary_segment_direct(sample3);
    MaskC valid = bss.is_valid;

    // _p0 on a face edge, _p2 on an emitter
    const Vector3fC &_p0    = detach(bss.p0);
    Vector3fC       &_p2    = bss.p2,
                    _dir    = normalize(_p2 - _p0);                    

    // check visibility between _p0 and _p2
    IntersectionC _its2;
    TriangleInfoD tri_info;
    // if constexpr ( ad ) {
        // _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
    // } else {
    _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid);
    // }
    valid &= (~_its2.is_valid());// && norm(_its2.p - _p2) < ShadowEpsilon;

    // trace another ray in the opposite direction to complete the boundary segment (_p1, _p2)
    // IntersectionD its4;
     IntersectionC _its1;
    if constexpr ( ad ) {
        _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), MaskC(valid), &tri_info);
        // its4 = scene.ray_intersect<true, true>(RayD(_p0, normalize(_p0 - bss.p2)), MaskD(valid));
    } else {
        _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), MaskC(valid));
    }
    // IntersectionC _its1 = detach(its1);
    valid &= _its1.is_valid();
    // std::cout<<" valid : "<<any(valid)<<std::endl;
    // Vector3fC &_p1 = _its1.p;

    // Start: Xi Deng added:  sample a point for bssrdf 
    BSDFSampleC bs1, bs2;
    BSDFArrayC bsdf_array = _its1.shape->bsdf(valid);
    bs1 = bsdf_array->sample(&scene, _its1, scene.m_samplers[2].next_nd<8, false>(), valid);
    Vector3fC &_p1 = bs1.po.p;
    
    // End: Xi Deng added
    bs2.po = _its1;
    bs2.wo = _its1.sh_frame.to_local(_dir);
    bs2.is_valid = _its1.is_valid();
    bs2.pdf = bss.pdf;
    bs2.is_sub = bsdf_array->hasbssdf();
    valid &= bs1.po.is_valid();
    // std::cout<<" valid : "<<any(valid)<<std::endl;

    // project _p1 onto the image plane and compute the corresponding pixel id
    SensorDirectSampleC sds = sensor.sample_direct(_p1);
    valid &= sds.is_valid;
    // std::cout<<" valid : "<<any(valid)<<std::endl;
    // trace a camera ray toward _p1 in a differentiable fashion
    Ray<ad> camera_ray;
    Intersection<ad> its2;
    if constexpr ( ad ) {
        camera_ray = sensor.sample_primary_ray(Vector2fD(sds.q));
        its2 = scene.ray_intersect<true, false>(camera_ray, valid);
        valid &= its2.is_valid() && norm(detach(its2.p) - _p1) < ShadowEpsilon;
    } else {
        camera_ray = sensor.sample_primary_ray(sds.q);
        its2 = scene.ray_intersect<false>(camera_ray, valid);
        valid &= its2.is_valid() && norm(its2.p - _p1) < ShadowEpsilon;
    }
    // std::cout<<" valid : "<<any(valid)<<std::endl;
    // calculate base_value
    FloatC      dist    = norm(_p2 - _its1.p),
                part_dist    = norm(_p2 - _p0);
                // cos2    = abs(dot(_its1.n, -_dir));
    Vector3fC   e       = cross(bss.edge, _dir); // perpendicular to edge and ray
    FloatC      sinphi  = norm(e);
    Vector3fC   proj    = normalize(cross(e, _its1.n)); // on the light plane
    FloatC      sinphi2 = norm(cross(_dir, proj));
    FloatC      base_v  = (dist/part_dist)*(sinphi/sinphi2);
    valid &= (sinphi > Epsilon) && (sinphi2 > Epsilon);
    // std::cout<<" valid : "<<any(valid)<<std::endl;
    // evaluate BSDF at _p1
    
    Vector3fC d0;
    if constexpr ( ad ) {
        d0 = -detach(camera_ray.d);
    } else {
        d0 = -camera_ray.d;
    }
    // std::cout<<" print direction : "<<d0<<std::endl;
    // std::cout<<" print frame : "<<bs1.po.sh_frame.n<<std::endl;
    // std::cout<<" print frame : "<<bs1.wo<<std::endl;

    bs1.wo =  bs1.po.sh_frame.to_local(d0);
    bs1.po.wi =  bs1.po.sh_frame.to_local(d0);
    SpectrumC bsdf_val = bsdf_array->eval(bs1.po, bs2, valid);
    FloatC pdfpoint = bsdf_array->pdfpoint(bs2.po, bs1, valid);
    SpectrumC value0 = (bsdf_val*scene.m_emitters[0]->eval(bs2.po, valid)*(base_v * sds.sensor_val/bss.pdf/pdfpoint)) & valid;
    if constexpr ( ad ) {
        Vector3fC n = normalize(cross(_its1.n, proj));
        value0 *= sign(dot(e, bss.edge2))*sign(dot(e, n));

        // Start Xi Deng modified
        const Vector3fD &v0 = tri_info.p0,
                        &e1 = tri_info.e1,
                        &e2 = tri_info.e2;

        Vector3fD myp2 = bss.p2 * 0.0f;
        set_slices(myp2, slices(bss.p0));
        myp2 = myp2 + bss.p2;
        Vector2fD uv;
        
        RayD shadow_ray(myp2, normalize(bss.p0 - myp2));
        std::tie(uv, std::ignore) = ray_intersect_triangle<true>(v0, e1, e2, shadow_ray);
        Vector3fD u2 = bilinear<true>(detach(v0), detach(e1), detach(e2), uv);
        // End Xi Deng added

        SpectrumD result = (SpectrumD(value0) * dot(Vector3fD(n), u2)) & valid;
        return { select(valid, sds.pixel_idx, -1), result - detach(result) };
    } else {
        return { -1, value0 };
    }
}

} // namespace psdr
