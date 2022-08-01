#include <optix.h>
#include <psdr/constants.h>
#include "psdr_cuda.h"

extern "C" {
__constant__ Params params;
}

/* Begin: Xi Deng added: per ray data from osc */
static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1 )
{
const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
void*           ptr = reinterpret_cast<void*>( uptr ); 
return ptr;
}

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
i0 = uptr >> 32;
i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD()
{ 
const uint32_t u0 = optixGetPayload_0();
const uint32_t u1 = optixGetPayload_1();
return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}
/* End: Xi Deng added */
  

extern "C" __global__ void __raygen__psdr_rg()
{
    const int tid = optixGetLaunchIndex().x;
    SurfacePos pos;
    uint32_t u0, u1;
    packPointer(&pos, u0, u1);
    pos.p_x = params.ray_o_x[tid];
    pos.p_y = params.ray_o_y[tid];
    pos.p_z = params.ray_o_z[tid];
    pos.t_max = params.ray_tmax[tid];
    pos.depth = 0;
    pos.layer_id = 1;
 
    if (params.rendermode == 0){
        // printf("layer id %d", pos.layer_id);
        while (pos.layer_id != -1 && pos.layer_id != 0){
            optixTrace(
                params.handle,// the accel we trace against
                make_float3(pos.p_x, pos.p_y, pos.p_z),
                make_float3(params.ray_d_x[tid], params.ray_d_y[tid], params.ray_d_z[tid]),
                psdr::RayEpsilon,
                pos.t_max,
                0.0f,
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0,
                1,
                0,
                u0, u1);
            if(pos.t_max < 0.0 || pos.layer_id == -2) break;
        }
    }// xi deng added for bssrdf intersections
    else{
        while (pos.depth < params.renderdepth && pos.layer_id != -2){
            optixTrace(
                params.handle,// the accel we trace against
                make_float3(pos.p_x, pos.p_y, pos.p_z),
                make_float3(params.ray_d_x[tid], params.ray_d_y[tid], params.ray_d_z[tid]),
                psdr::RayEpsilon,
                pos.t_max,
                0.0f,
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0, // sbt offset
                1, // sbt stride
                0, // misssbtindex
                u0, u1); // per ray data 
            if ( pos.t_max < 0.0 || pos.layer_id == -2) break;
        }
        
        // sample data from intersections uniformly
        if (pos.depth > 0){
            uint32_t sampleIdx = tid * params.renderdepth;

            if(params.rendermode == 1){
                const uint32_t idx          = floor(params.rnd[ tid ] * pos.depth);
                sampleIdx    = tid * params.renderdepth + idx;
            }
            params.tri_index[ tid ]     = params.mul_tri_index[ sampleIdx ];
            params.shape_index[ tid ]   = params.mul_shape_index[ sampleIdx ];
            params.barycentric_u[ tid ] = params.mul_barycentric_u[ sampleIdx ];
            params.barycentric_v[ tid ] = params.mul_barycentric_v[ sampleIdx ];
            params.num_its[ tid ]       = pos.depth;
        }else{
            params.tri_index[ tid ]     = -1;
            params.shape_index[ tid ]   = -1;
            params.barycentric_u[ tid ] = -1.0f;
            params.barycentric_v[ tid ] = -1.0f;
            params.num_its[ tid ]         = 0;
        }
        
    }
}

extern "C" __global__ void __miss__psdr_ms()
{   
    const uint32_t image_index          = optixGetLaunchIndex().x;
    params.tri_index[ image_index ]     = -1;
    params.shape_index[ image_index ]   = -1;
    params.barycentric_u[ image_index ] = -1.0f;
    params.barycentric_v[ image_index ] = -1.0f;
    if (params.rendermode == 0){
       
        SurfacePos &prd = *(SurfacePos*)getPRD<SurfacePos>();
        prd.p_x = 0.0f;
        prd.p_y = 0.0f;
        prd.p_z = 0.0f;
        prd.t_max = -1.0f;
        prd.layer_id = -2;

    } /* xi deng added for supporting bssrdf */
    else if(params.rendermode > 0){  
        SurfacePos &prd = *(SurfacePos*)getPRD<SurfacePos>();
        uint32_t offsetIdx = image_index * params.renderdepth;
            
       
        for (int i = prd.depth ; i < params.renderdepth ; i++) {
            params.mul_tri_index[ offsetIdx + i ]     = -1;
            params.mul_shape_index[ offsetIdx + i ]   = -1;
            params.mul_barycentric_u[ offsetIdx + i ] = -1.0f;
            params.mul_barycentric_v[ offsetIdx + i ] = -1.0f;
        }    

        //  if (image_index == 0)
            // printf("offsetIdx %d, prd.depth %d, uv %f \n", image_index, prd.depth, params.mul_barycentric_u[ 0 ]);  
        
        prd.p_x = 0.0f;
        prd.p_y = 0.0f;
        prd.p_z = 0.0f;
        prd.t_max = -1.0f;
        prd.layer_id = -2;
    }
}
 
extern "C" __global__ void __closesthit__psdr_ch()
{   

    SurfacePos &prd = *(SurfacePos*)getPRD<SurfacePos>();
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    int primitiveID = optixGetPrimitiveIndex();

    const uint32_t image_index          = optixGetLaunchIndex().x;
    int3 indices    = make_int3(rt_data->index[primitiveID * 3], rt_data->index[primitiveID * 3 + 1], rt_data->index[primitiveID * 3 + 2]);
    float3 v0       = make_float3(rt_data->vertex[indices.x * 3], rt_data->vertex[indices.x * 3 + 1], rt_data->vertex[indices.x * 3 + 2]);
    float3 v1       = make_float3(rt_data->vertex[indices.y * 3], rt_data->vertex[indices.y * 3 + 1], rt_data->vertex[indices.y * 3 + 2]);
    float3 v2       = make_float3(rt_data->vertex[indices.z * 3], rt_data->vertex[indices.z * 3 + 1], rt_data->vertex[indices.z * 3 + 2]);
    float2 uv       = optixGetTriangleBarycentrics();
    float3 surpos   = make_float3(  (1.f - uv.x - uv.y) * v0.x + uv.x  * v1.x + uv.y  * v2.x,
                                        (1.f - uv.x - uv.y) * v0.y + uv.x  * v1.y + uv.y  * v2.y,
                                        (1.f - uv.x - uv.y) * v0.z + uv.x  * v1.z + uv.y  * v2.z);
    // get traveled distance
    float3 dist     = make_float3(surpos.x - prd.p_x , surpos.y - prd.p_y, surpos.z - prd.p_z);
    prd.p_x         = surpos.x;
    prd.p_y         = surpos.y;
    prd.p_z         = surpos.z;
    prd.t_max       = prd.t_max - sqrt(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
    prd.layer_id    = rt_data->layer_id;

    // check if hit the outer surface
    if (prd.layer_id < 1){
        if (params.rendermode  == 0){
            params.tri_index[ image_index ]     = primitiveID + rt_data->shape_offset;
            params.shape_index[ image_index ]   = rt_data->shape_id;
            params.barycentric_u[ image_index ] = uv.x;
            params.barycentric_v[ image_index ] = uv.y;
        } /* xi deng added for supporting bssrdf */
        else if(params.rendermode == 1){
        /* Xi Deng added */
            const uint32_t individual_index = image_index * params.renderdepth + prd.depth;
            // write to large array
            params.mul_tri_index[ individual_index ] = primitiveID + rt_data->shape_offset;
            params.mul_shape_index[ individual_index ] = rt_data->shape_id;
            params.mul_barycentric_u[ individual_index ] = uv.x;
            params.mul_barycentric_v[ individual_index ] = uv.y; 
            // printf(" mode 1: uv [%f,%f] \n", uv.x, uv.y);
            prd.depth       += 1;
        }
    } else {// if hit the inner surface
        if (params.rendermode == 2){
        /* Xi Deng added for multilayer uv test */
            const uint32_t individual_index = image_index * params.renderdepth + prd.depth;
            // if (image_index == 0)
                // printf("hit something unexpectedly!");
            params.mul_tri_index[ individual_index ] = primitiveID + rt_data->shape_offset;
            params.mul_shape_index[ individual_index ] = rt_data->shape_id;
            params.mul_barycentric_u[ individual_index ] = uv.x;
            params.mul_barycentric_v[ individual_index ] = uv.y;       
            // printf(" mode 2: idx [%d] uv [%f,%f] \n", individual_index, uv.x, uv.y);
            // printf(" mode 2: idx [%d] uv [%f,%f] \n", individual_index, params.mul_barycentric_u[individual_index], params.mul_barycentric_v[ individual_index ]);
            prd.depth       += 1;
        }
    }
}
