#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/scene/optix.h>
#include <psdr/scene/scene_optix.h>
#include <psdr/shape/mesh.h>

namespace psdr
{

void Intersection_OptiX::reserve(int64_t size, int64_t depth) {
    PSDR_ASSERT(size > 0);
    if ( size != m_size || m_depth != depth) {
        m_size = size;
        m_depth = depth;
        // xi deng muldified for multilayer
        triangle_id = empty<IntC>(size);
        shape_id = empty<IntC>(size);
        uv = empty<Vector2fC>(size);

        numIntersections = empty<IntC>(size);
        if (depth != 0){
            m_depth = depth;
            triangle_ids = empty<IntC>(size * depth);
            shape_ids = empty<IntC>(size * depth);
            uvs = empty<Vector2fC>(size * depth);
        }
    }
}


Scene_OptiX::Scene_OptiX() {
    m_accel = nullptr;
}


Scene_OptiX::~Scene_OptiX() {
    if ( m_accel != nullptr ) {
        optix_release(*m_accel);
        delete m_accel;
    }
}


void Scene_OptiX::configure(const std::vector<Mesh *> &meshes) {
    PSDR_ASSERT(!meshes.empty());
    size_t num_meshes = meshes.size();

    if ( m_accel != nullptr ) {
        /* xi deng revised to support bssrdf */
        optix_release(*m_accel);
        /* origin code */

        // std::vector<int> face_offset(num_meshes + 1);
        // face_offset[0] = 0;
        // for ( size_t i = 0; i < num_meshes; ++i )
        //     face_offset[i + 1] = face_offset[i] + meshes[i]->m_num_faces;
        // m_accel = new PathTracerState();
        // optix_config(*m_accel, face_offset);
    }

    uint32_t triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    std::vector<CUdeviceptr> vertex_buffer_ptrs(num_meshes);
    std::vector<OptixBuildInput> build_inputs(num_meshes);
    // m_buildInput.resize(num_meshes)

    /*BEGIN: xi deng added */
    std::vector<FaceInfo> face_information(num_meshes+1);
    face_information[0].face_offset = 0;

    /*END: xi deng added*/
    for ( size_t i = 0; i < num_meshes; ++i ) {
        /*BEGIN: xi deng added */
        face_information[i+1].face_offset = face_information[i].face_offset + meshes[i]->m_num_faces;
        face_information[i].layer_id = meshes[i]->m_layer_count;
        /*END: xi deng added*/

        const Mesh *mesh = meshes[i];

        PSDR_ASSERT(static_cast<int>(slices(mesh->m_vertex_buffer)) == mesh->m_num_vertices*3);
        PSDR_ASSERT(static_cast<int>(slices(mesh->m_face_buffer)) == mesh->m_num_faces*3);

        build_inputs[i].type                                = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build_inputs[i].triangleArray.vertexFormat          = OPTIX_VERTEX_FORMAT_FLOAT3;
        build_inputs[i].triangleArray.vertexStrideInBytes   = 3*sizeof(float);
        build_inputs[i].triangleArray.numVertices           = static_cast<uint32_t>(slices(mesh->m_vertex_buffer)/3);
        vertex_buffer_ptrs[i]                               = reinterpret_cast<CUdeviceptr>(mesh->m_vertex_buffer.data());
        build_inputs[i].triangleArray.vertexBuffers         = &vertex_buffer_ptrs[i];

        build_inputs[i].triangleArray.indexFormat           = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        build_inputs[i].triangleArray.numIndexTriplets      = static_cast<uint32_t>(slices(mesh->m_face_buffer)/3);
        build_inputs[i].triangleArray.indexBuffer           = reinterpret_cast<CUdeviceptr>(mesh->m_face_buffer.data());
        build_inputs[i].triangleArray.indexStrideInBytes    = 3*sizeof(int);

        build_inputs[i].triangleArray.flags                 = triangle_input_flags;
        build_inputs[i].triangleArray.numSbtRecords         = 1;

        face_information[i].vertex = (float *)reinterpret_cast<CUdeviceptr>(mesh->m_vertex_buffer.data());
        face_information[i].index  = (int *)reinterpret_cast<CUdeviceptr>(mesh->m_face_buffer.data());
    }

    // if (m_accel == nullptr){
    m_accel = new PathTracerState();
    optix_config(*m_accel, face_information);
    // }

    build_accel(*m_accel, build_inputs);
}


bool Scene_OptiX::is_ready() const {
    return m_accel != nullptr;
}

// xi deng added for bssrdf / should add another multilayer version
template <bool ad>
Vector2i<ad> Scene_OptiX::ray_all_layer(const Ray<ad> &ray, Mask<ad> &active, const int depth) const {
    const int m = static_cast<int>(slices(ray.o));
    m_its.reserve(m, depth);

    cuda_eval();

    m_accel->params.ray_o_x             = ray.o.x().data();
    m_accel->params.ray_o_y             = ray.o.y().data();
    m_accel->params.ray_o_z             = ray.o.z().data();

    m_accel->params.ray_d_x             = ray.d.x().data();
    m_accel->params.ray_d_y             = ray.d.y().data();
    m_accel->params.ray_d_z             = ray.d.z().data();

    m_accel->params.ray_tmax            = ray.tmax.data();
    m_accel->params.tri_index           = m_its.triangle_id.data();
    m_accel->params.shape_index         = m_its.shape_id.data();
    m_accel->params.barycentric_u       = m_its.uv.x().data();
    m_accel->params.barycentric_v       = m_its.uv.y().data();

    m_accel->params.mul_tri_index       = m_its.triangle_ids.data();
    m_accel->params.mul_shape_index     = m_its.shape_ids.data();
    m_accel->params.mul_barycentric_u   = m_its.uvs.x().data();
    m_accel->params.mul_barycentric_v   = m_its.uvs.y().data();

    m_accel->params.rendermode          = 2;
    m_accel->params.renderdepth         = depth;
    m_accel->params.rnd                 = nullptr;
    m_accel->params.num_its             = m_its.numIntersections.data();
    m_accel->params.maxRange            = m * depth;


    CUDA_CHECK(
        cudaMemcpyAsync(
            reinterpret_cast<void*>( m_accel->d_params ),
            &m_accel->params, sizeof( Params ),
            cudaMemcpyHostToDevice, m_accel->stream
        )
    );


    OPTIX_CHECK(
        optixLaunch(
            m_accel->pipeline,
            m_accel->stream,
            reinterpret_cast<CUdeviceptr>( m_accel->d_params ),
            sizeof( Params ),
            &m_accel->sbt,
            m,              // launch size
            1,              // launch height
            1               // launch depth
        )
    );

    CUDA_SYNC_CHECK();

    // std::cout<<"uvs "<<m_its.uvs<<std::endl;
    // std::cout<<"uvs "<<m_its.uv<<std::endl;

    // std::cout<<"c1: m_its.shape_ids slices"<<slices(m_its.shape_ids)<<std::endl;
    active &= (m_its.shape_ids >= 0) && (m_its.triangle_ids >= 0);
    // std::cout<<"c2: m_its.shape_ids slices"<<slices(m_its.shape_ids)<<std::endl;

    return Vector2i<ad>(m_its.shape_ids, m_its.triangle_ids);
}

template <bool ad>
Vector3i<ad> Scene_OptiX::ray_all_intersect(const Ray<ad> &ray, Mask<ad> &active, const Vector8f<ad> &sample, const int depth) const {
    const int m = static_cast<int>(slices(ray.o));
    m_its.reserve(m);

    IntC triangle_ids   = empty<IntC>(m * depth);
    IntC shape_ids      = empty<IntC>(m * depth);
    Vector2fC uvs       = empty<Vector2fC>(m * depth);

    cuda_eval();

    m_accel->params.ray_o_x         = ray.o.x().data();
    m_accel->params.ray_o_y         = ray.o.y().data();
    m_accel->params.ray_o_z         = ray.o.z().data();

    m_accel->params.ray_d_x         = ray.d.x().data();
    m_accel->params.ray_d_y         = ray.d.y().data();
    m_accel->params.ray_d_z         = ray.d.z().data();

    m_accel->params.ray_tmax        = ray.tmax.data();
    m_accel->params.tri_index       = m_its.triangle_id.data();
    m_accel->params.shape_index     = m_its.shape_id.data();
    m_accel->params.barycentric_u   = m_its.uv.x().data();
    m_accel->params.barycentric_v   = m_its.uv.y().data();

    m_accel->params.mul_tri_index   = triangle_ids.data();
    m_accel->params.mul_shape_index = shape_ids.data();
    m_accel->params.mul_barycentric_u = uvs.x().data();
    m_accel->params.mul_barycentric_v = uvs.y().data();

    m_accel->params.rendermode      = 1;
    m_accel->params.renderdepth     = depth;
    m_accel->params.rnd             = sample[4].data();
    m_accel->params.num_its         = m_its.numIntersections.data();
    m_accel->params.maxRange        = m * depth;


    CUDA_CHECK(
        cudaMemcpyAsync(
            reinterpret_cast<void*>( m_accel->d_params ),
            &m_accel->params, sizeof( Params ),
            cudaMemcpyHostToDevice, m_accel->stream
        )
    );


    OPTIX_CHECK(
        optixLaunch(
            m_accel->pipeline,
            m_accel->stream,
            reinterpret_cast<CUdeviceptr>( m_accel->d_params ),
            sizeof( Params ),
            &m_accel->sbt,
            m,              // launch size
            1,              // launch height
            1               // launch depth
        )
    );

    CUDA_SYNC_CHECK();


    active &= (m_its.shape_id >= 0) && (m_its.triangle_id >= 0);
    return Vector3i<ad>(m_its.shape_id, m_its.triangle_id, m_its.numIntersections);

}

template <bool ad>
Vector2i<ad> Scene_OptiX::ray_intersect(const Ray<ad> &ray, Mask<ad> &active) const {
    const int m = static_cast<int>(slices(ray.o));
    m_its.reserve(m);

    cuda_eval();

    m_accel->params.ray_o_x         = ray.o.x().data();
    m_accel->params.ray_o_y         = ray.o.y().data();
    m_accel->params.ray_o_z         = ray.o.z().data();

    m_accel->params.ray_d_x         = ray.d.x().data();
    m_accel->params.ray_d_y         = ray.d.y().data();
    m_accel->params.ray_d_z         = ray.d.z().data();

    m_accel->params.ray_tmax        = ray.tmax.data();
    m_accel->params.tri_index       = m_its.triangle_id.data();
    m_accel->params.shape_index     = m_its.shape_id.data();
    m_accel->params.barycentric_u   = m_its.uv.x().data();
    m_accel->params.barycentric_v   = m_its.uv.y().data();
    
    m_accel->params.maxRange        = m;

    m_accel->params.rendermode      = 0;

    CUDA_CHECK(
        cudaMemcpyAsync(
            reinterpret_cast<void*>( m_accel->d_params ),
            &m_accel->params, sizeof( Params ),
            cudaMemcpyHostToDevice, m_accel->stream
        )
    );

    OPTIX_CHECK(
        optixLaunch(
            m_accel->pipeline,
            m_accel->stream,
            reinterpret_cast<CUdeviceptr>( m_accel->d_params ),
            sizeof( Params ),
            &m_accel->sbt,
            m,              // launch size
            1,              // launch height
            1               // launch depth
        )
    );

    CUDA_SYNC_CHECK();

    active &= (m_its.shape_id >= 0) && (m_its.triangle_id >= 0);
    return Vector2i<ad>(m_its.shape_id, m_its.triangle_id);
}


// Explicit instantiations
template Vector2iC Scene_OptiX::ray_all_layer(const RayC &ray, MaskC &active, const int depth) const;
template Vector2iD Scene_OptiX::ray_all_layer(const RayD &ray, MaskD &active, const int depth) const;

template Vector3iC Scene_OptiX::ray_all_intersect(const RayC &ray, MaskC &active, const Vector8fC &sample, const int depth) const;
template Vector3iD Scene_OptiX::ray_all_intersect(const RayD &ray, MaskD &active, const Vector8fD &sample, const int depth) const;

template Vector2iC Scene_OptiX::ray_intersect(const RayC &ray, MaskC &active) const;
template Vector2iD Scene_OptiX::ray_intersect(const RayD &ray, MaskD &active) const;

} // namespace psdr
