#include "AllIntersectRenderer.h"

#include <optix_function_table_definition.h>

namespace dx {
    extern "C" char embedded_ptx_code[];

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *data;
    };

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *data;
    };

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        int objectID;
    };

    AllIntersectRenderer::AllIntersectRenderer()
    {
        initOptix();
        createContext();
        createModule();
        createRaygenPrograms();
        createMissPrograms();
        createHitgraoupProgram();
        createPipline();
        buildSBT();

        launchParamsBuffer.alloc(sizeof(launchParams));

    }

    void AllIntersectRenderer::initOptix(){
         std::cout << "#osc: initializing optix..." << std::endl;
      
        // -------------------------------------------------------
        // check for available optix7 capable devices
        // -------------------------------------------------------
        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0)
        throw std::runtime_error("#osc: no CUDA capable devices found!");
        std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

        // -------------------------------------------------------
        // initialize optix
        // -------------------------------------------------------
        OPTIX_CHECK( optixInit() );
        std::cout << "#osc: successfully initialized optix... yay! " << std::endl;
    }

      static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
    {
        fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
    }

    void AllIntersectRenderer::createContext(){
        const int deviceID = 0;
        CUDA_CHECK(SetDevice(deviceID));
        CUDA_CHECK(StreamCreate(&stream));

        cudaGetDeviceProperties(&deviceProps, deviceID);
        std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

        // GET CURRENT CUDA DEVICE CONTEXT
        CUresult cuRes = cuCtxGetCurrent(&cudaContext);
        if( cuRes != CUDA_SUCCESS )
            fprintf( stderr, "Error querying current context: error code %d\n", cuRes );

        // CREATE OPTIX CONTEXT
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
        OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
    }

    void AllIntersectRenderer::createModule()
    {   
        // COMPILEOPTIONS
        moduleCompileOptions.maxRegisterCount   = 50;
        moduleCompileOptions.optLevel           = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel         = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pipelineCompileOptions                          = {};
        pipelineCompileOptions.traversableGraphFlags    = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur           = false;
        pipelineCompileOptions.numPayloadValues         = 2;
        pipelineCompileOptions.numAttributeValues       = 2;
        pipelineCompileOptions.exceptionFlags           = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";

        // LINK OPTION
        pipelineLinkOptions.maxTraceDepth               = 2;

        const std::string ptxCode = embedded_ptx_code;

        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                            &moduleCompileOptions,
                                            &pipelineCompileOptions,
                                            ptxCode.c_str(),
                                            ptxCode.size(),
                                            log,&sizeof_log,
                                            &module
                                            ));
        if ( sizeof_log > 1 ) PRINT(log);
    }

    void AllIntersectRenderer::createRaygenPrograms()
    {
        // set up program group specification;
        raygenPGs.resize(1);
        OptixProgramGroupOptions pgOptions  = {};
        OptixProgramGroupDesc pgDesc        = {};
        pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.miss.module                  = module;
        pgDesc.miss.entryFunctionName       = "__raygen__renderFrame";

        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &raygenPGs[0]
                                            ));
        if (sizeof_log > 1) PRINT(log);
    }

    void AllIntersectRenderer::createMissPrograms()
    {
        missPGs.resize(1);

        OptixProgramGroupOptions pgOptions  = {};
        OptixProgramGroupDesc pgDesc        = {};
        pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module                  = module;
        pgDesc.miss.entryFunctionName       = "__miss__radiance";

        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &missPGs[0]
                                            ));
        if (sizeof_log > 1) PRINT(log);
    }

    void AllIntersectRenderer::createHitgraoupProgram()
    {
        hitgroupPGs.resize(1);

        OptixProgramGroupOptions pgOptions  = {};
        OptixProgramGroupDesc pgDesc        = {};
        pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH            = module;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        pgDesc.hitgroup.moduleAH            = module;
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &hitgroupPGs[0]
                                            ));
        if (sizeof_log > 1) PRINT(log);
    }

    void AllIntersectRenderer::createPipline()
    {
        std::vector<OptixProgramGroup> programGroups;
        for (auto pg : raygenPGs)
            programGroups.push_back(pg);
        for (auto pg : missPGs)
            programGroups.push_back(pg);
        for (auto pg : hitgroupPGs)
            programGroups.push_back(pg);

        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixPipelineCreate(optixContext,
                                        &pipelineCompileOptions,
                                        &pipelineLinkOptions,
                                        programGroups.data(),
                                        (int) programGroups.size(),
                                        log, &sizeof_log,
                                        &pipeline
                                        ));
        if ( sizeof_log > 1 ) PRINT(log);

        OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
                                                2*1024,
                                                2*1024,
                                                2*1024,
                                                1
                                                ));
        if (sizeof_log > 1 ) PRINT(log);
    }

    // CONSTRUCT SHADER BINDING TABLE
    void AllIntersectRenderer::buildSBT()
    {
        std::vector<RaygenRecord> raygenRecords;
        for (int i = 0; i<raygenPGs.size();i++){
            RaygenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
            rec.data = nullptr;
            raygenRecords.push_back(rec);
        }
        raygenRecordsBuffer.alloc_and_upload(raygenRecords);
        sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

        std::vector<MissRecord> missRecords;
        for (int i = 0 ; i < missPGs.size(); i++){
            MissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
            rec.data = nullptr;
            missRecords.push_back(rec);
        }
        missRecordsBuffer.alloc_and_upload(missRecords);
        sbt.missRecordBase          = missRecordsBuffer.d_pointer();
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount         = (int)missRecords.size();

        int numObjects = 1;
        std::vector<HitGroupRecord> hitgroupRecords;
        for (int i = 0 ; i < numObjects ; i++) {
            int objectType = 0;
            HitGroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
            rec.objectID = i;
            hitgroupRecords.push_back(rec);
        }
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();

    }

    void AllIntersectRenderer::render()
    {
        if (launchParams.sizeX == 0) return;

        launchParamsBuffer.upload(&launchParams, 1);
        launchParams.frameID++;

        OPTIX_CHECK(optixLaunch(
                                pipeline, stream,
                                launchParamsBuffer.d_pointer(),
                                launchParamsBuffer.sizeInBytes,
                                &sbt,
                                launchParams.sizeX,
                                launchParams.sizeY,
                                1
                                ));

        CUDA_SYNC_CHECK();
    }

    void AllIntersectRenderer::resize(const int sX, const int sY){
        if (sX == 0 | sY == 0) return;

        colorBuffer.resize(sX * sY * sizeof(uint32_t));

        launchParams.sizeX = sX;
        launchParams.sizeY = sY;

        launchParams.colorBuffer = (uint32_t*)colorBuffer.d_ptr;
    }

    void AllIntersectRenderer::downloadPixels(uint32_t h_pixels[]){
        colorBuffer.download(h_pixels, launchParams.sizeX * launchParams.sizeY);
    }
}