#pragma once

#include <stdint.h>

namespace dx {
    struct LaunchParams {

        uint32_t sizeX;
        uint32_t sizeY;
        uint32_t *colorBuffer;
        int frameID { 0 };

        struct {
            float p_x, p_y, p_z;
            float d_x, d_y, d_z;
            float h_x, h_y, h_z;
            float v_x, v_y, v_z;
        } camera;

        OptixTraversableHandle traversable;
    };
} // ::dx