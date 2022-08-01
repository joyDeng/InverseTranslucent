#include <stdint.h>

struct Params
{
    const float  *ray_o_x, *ray_o_y, *ray_o_z;
    const float  *ray_d_x, *ray_d_y, *ray_d_z;
    const float  *ray_tmax;

    int          *tri_index;
    int          *shape_index;
    float        *barycentric_u, *barycentric_v;

    /* xi deng added for bssrdf intersections */
    int *mul_tri_index;
    int *mul_shape_index;
    float *mul_barycentric_u, *mul_barycentric_v;
    
    //float *mul_layer_index; 
    int *num_its;

    /* xi deng added for bssrdf intersections */
    /* render mode 0, return closest intersection */
    /* render mode 1, bssrdf point sample, return all intersections */
    /* TODO: render mode 2, multilayer bssrdf, return intersections with all layers */
    int rendermode;

    int renderdepth;
    const float *rnd;
    uint32_t maxRange;


    OptixTraversableHandle handle;
};

struct RayGenData
{

};

struct MissData
{

};

struct HitGroupData
{
	int shape_offset;
	int shape_id;
    float *vertex;
    int *index;
    int layer_id;
};

typedef struct SurfacePos
{
    float p_x, p_y, p_z;
    float t_max;
    int depth;
    int layer_id;
} SurfacePos;
