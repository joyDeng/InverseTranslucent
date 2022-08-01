#pragma once

namespace psdr
{
    // Core classes

    template <typename> struct Frame_;
    template <bool ad>
    using Frame = Frame_<Float<ad>>;
    using FrameC = Frame<false>;
    using FrameD = Frame<true>;

    template <bool ad> struct Ray;
    using RayC = Ray<false>;
    using RayD = Ray<true>;

    template <typename> struct Interaction_;

    template <bool ad>
    using Interaction  = Interaction_<Float<ad>>;
    using InteractionC = Interaction<false>;
    using InteractionD = Interaction<true>;
    
    

    struct Intersection_OptiX;
    struct LayerIntersection;

    // template <bool ad> struct Intersection;
    template <typename> struct Intersection_;

    template <bool ad>
    using Intersection  = Intersection_<Float<ad>>;

    using IntersectionC = Intersection<false>;
    using IntersectionD = Intersection<true>;
    
    template <bool ad>
    using IntersectionArray = Array<Intersection<ad>, 20>;

    // using IntersectionArray  = Type<Intersection*, ad>;
    // using IntersectionArrayC = IntersectionArray<false>;
    // using IntersectionArrayD = IntersectionArray<true>;
    // using IntersectionVector = Intersections__<Float<ad>>;

    struct Sampler;

    struct DiscreteDistribution;

    template <int> struct HyperCubeDistribution;
    using HyperCubeDistribution2f = HyperCubeDistribution<2>;
    using HyperCubeDistribution3f = HyperCubeDistribution<3>;

    struct MicrofacetDistribution;

    // Sampling records

    template <typename> struct SampleRecord_;
    template <bool ad>
    using SampleRecord          = SampleRecord_<Float<ad>>;
    using SampleRecordC         = SampleRecord<false>;
    using SampleRecordD         = SampleRecord<true>;

    template <typename> struct DirectionSample_;
    template <bool ad>
    using DirectionSample       = DirectionSample_<Float<ad>>;
    using DirectionSampleC      = DirectionSample<false>;
    using DirectionSampleD      = DirectionSample<true>;

    template <typename> struct PositionSample_;
    template <bool ad>
    using PositionSample        = PositionSample_<Float<ad>>;
    using PositionSampleC       = PositionSample<false>;
    using PositionSampleD       = PositionSample<true>;

    template <typename> struct BSDFSample_;
    template <bool ad>
    using BSDFSample            = BSDFSample_<Float<ad>>;
    using BSDFSampleC           = BSDFSample<false>;
    using BSDFSampleD           = BSDFSample<true>;

// Xi Deng added
    // template <typename> struct BSSDFSample_;
    // template <bool ad>
    // using BSSDFSample            = BSSDFSample_<Float<ad>>;
    // using BSSDFSampleC           = BSSDFSample<false>;
    // using BSSDFSampleD           = BSSDFSample<true>;
// Xi Deng added    

    template <typename> struct SensorDirectSample_;
    template <bool ad>
    using SensorDirectSample    = SensorDirectSample_<Float<ad>>;
    using SensorDirectSampleC   = SensorDirectSample<false>;
    using SensorDirectSampleD   = SensorDirectSample<true>;

    struct BoundarySegSampleDirect;

    // Main classes
    class BSDF;
    template <bool ad>
    using BSDFArray     = Type<BSDF*, ad>;
    using BSDFArrayC    = BSDFArray<false>;
    using BSDFArrayD    = BSDFArray<true>;

    class Diffuse;
    class RoughConductor;
    // class Subsurface;

    class Emitter;
    template <bool ad>
    using EmitterArray  = Type<Emitter*, ad>;
    using EmitterArrayC = EmitterArray<false>;
    using EmitterArrayD = EmitterArray<true>;

    class AreaLight;
    class EnvironmentMap;

    class Sensor;
    class PerspectiveCamera;

    class Mesh;
    template <bool ad>
    using MeshArray     = Type<Mesh*, ad>;
    using MeshArrayC    = MeshArray<false>;
    using MeshArrayD    = MeshArray<true>;

    class Integrator;
    class FieldExtractionIntegrator;
    class DirectIntegrator;
    class LaserIntegrator;
    class ColocateIntegrator;
    class OldDirectIntegrator;
    class Integrator;

    class Scene_OptiX;
    class Scene;
    class SceneLoader;
}
