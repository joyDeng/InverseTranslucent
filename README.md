# psdr-cuda
This is a path space differentiable renderer (psdr-cuda) with bssrdf support which forked from [psdr-cuda](https://psdr-cuda.readthedocs.io/en/latest/).


## Compile
To build the code, please follow this process [psdr-cuda](https://psdr-cuda.readthedocs.io/en/latest/core_compile.html)

## Materials
The material we use in reconstructing is `HeterSub`, the definition and implementation can be found in `hetersub.h` and `hetersub.cpp`, currently, it is using dipole model, one can switch to better diphole in the future.

## Integrators

Here are several integrators implemented in this framework, the one we used for reconstruction is the `DirectIntegrator`; the `OldDirectIntegrator` correspond to the `DirectIntegrator` in the original psdr-cuda codebase.

```c++
DirectIntegrator
```

Direct integrator with point light assumption, support gradient estimation of geometry with bssrdf material (both primary and secondary discontintuity).

```c++
ColocateIntegrator
```
This is integrator with the "point light locates at the same place as camera" assumpition.

```c++
LaserIntegrator
```
This is only used in bssrdf validation, where a directional coherent light is pointing down to a plane. Deprecated.

```c++
OldDirectIntegrator
```
This is the origin direct integrator in psdr-cuda, which support bsdf with MIS direct lighting.

## Optimizing

#### Real data 
The latest optimizing code is in "examples/python/learn_real_data.py"

To set the ROOT_DIR, SCENE_DIR, etc... go to "examples/python/constants.py"

To optimize homogeneous value of alebdo and sigma_t, use args.sigma_texture = 0

To optimize albedo and sigma_t as texture, use args.sigma_texture = 512 and args.albedo_texture = 512: 


## Data Folder
Find real data here [Soap and Kiwi](https://drive.google.com/drive/folders/1JrTtno7c-FnYuNJ044FKbjlZYujJiczN?usp=sharing).




