#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <string>
#include <assert.h>
#include <math.h>
 
 namespace dx {
    class AllIntersectRenderer
    {
        public:

        AllIntersectRenderer();

        void render();

        void resize(const int x,const int y);

        void downloadPixels(uint32_t h_pixels[]);

        protected:

        void initOptix();

        void createContext();

        void createModule();

        void createRaygenPrograms();

        void createMissPrograms();

        void createHitgraoupProgram();

        void createPipline();

        void buildSBT();

        protected:
        CUcontext       cudaContext;
        CUstream        stream;
        cudaDeviceProp  deviceProps;

        OptixDeviceContext optixContext;

        OptixPipeline               pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions  = {};
        OptixPipelineLinkOptions    pipelineLinkOptions     = {};

        OptixModule                 module;
        OptixModuleCompileOptions   moduleCompileOptions = {};

        std::vector<OptixProgramGroup> raygenPGs;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        CUDABuffer hitgroupRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        LaunchParams launchParams;
        CUDABuffer launchParamsBuffer;

        CUDABuffer colorBuffer;
    };
 } // :: dx