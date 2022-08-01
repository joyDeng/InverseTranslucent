#pragma once

#include <psdr/psdr.h>

struct PathTracerState;

namespace psdr
{

struct Intersection_OptiX {
    void reserve(int64_t size, int64_t depth=0);

    int64_t m_size = 0;
    int64_t m_depth = 0;
    IntC triangle_id;
    IntC shape_id;
    Vector2fC uv;
    //* xi deng added
    IntC numIntersections;
    IntC triangle_ids;
    IntC shape_ids;
    Vector2fC uvs;

};


class Scene_OptiX {
    friend class Scene;
    //friend std::unique_ptr<Scene_OptiX> std::make_unique<Scene_OptiX>();

public:
    ~Scene_OptiX();

protected:
    Scene_OptiX();
    void configure(const std::vector<Mesh*> &meshes);
    bool is_ready() const;

    template <bool ad>
    Vector2i<ad> ray_intersect(const Ray<ad> &ray, Mask<ad> &active) const;

    template <bool ad>
    Vector3i<ad> ray_all_intersect(const Ray<ad> &ray, Mask<ad> &active, const Vector8f<ad> &sample, const int depth) const;

    template <bool ad>
    Vector2i<ad> ray_all_layer(const Ray<ad> &ray, Mask<ad> &active, const int depth) const;

    PathTracerState                 *m_accel;
    mutable Intersection_OptiX      m_its;
    // std::vector<OptixBuildInput>    m_buildInput;
    // std::vector<CUdeviceptr> vertex_buffer_ptrs(num_meshes);
};

} // namespace psdr
