#include <optix_device.h>
#include "LaunchParams.h"


using namespace dx;

namespace dx{

    
    //* user specified launchparam struct
   extern "C" __constant__ LaunchParams launchParams;

   static __forceinline__ __device__
    void *unpackPointer( uint32_t i0, uint32_t i1 )
    {
        const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
        void*           ptr = reinterpret_cast<void*>( uptr );
        return ptr;
    }

    static __forceinline__ __device__
    void packPointer( void* ptr, uint32_t& i0, uint32_t& i1)
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
        return reinterpret_cast<T*>( unpackPointer( u0, u1 ));
    }


   extern "C" __global__ void __miss__radiance(){ 
       uint32_t &prd = *(uint32_t*)getPRD<uint32_t>();
       prd = 0xffffffff;
    }

    extern "C" __global__ void __anyhit__radiance(){ /*! for this simple example, this will remain empty */ }


   extern "C" __global__ void __closesthit__radiance(){ 
       const int primID = optixGetPrimitiveIndex();
       uint32_t &prd = *(uint32_t*)getPRD<uint32_t>();
       const int r = primID % 256;
       prd = 0xff000000 | (r << 0);
    }

   extern "C" __global__ void __raygen__renderFrame()
   {
       const int ix = optixGetLaunchIndex().x;
       const int iy = optixGetLaunchIndex().y;

       const auto &camera = launchParams.camera;

       uint32_t rgba;
       uint32_t u0, u1;
       packPointer( &rgba, u0, u1);

       

       const uint32_t rgba = 0xff000000 | (r<<0) | (g<<8) | (b<<16);

       const uint32_t fbIndex = ix+iy*launchParams.sizeX;
       launchParams.colorBuffer[fbIndex] = rgba;
   }
}