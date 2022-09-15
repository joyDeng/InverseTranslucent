# psdr-cuda
Path-space differentiable renderer (psdr-cuda) with bssrdf support.

## Compile
To build the code, please follow the process from [psdr-cuda](https://psdr-cuda.readthedocs.io/en/latest/core_compile.html)

## Integrator

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

To optimize alebdo and sigma_t as texture, use args.sigma_texture = 512 and args.albedo_texture = 512: 


## Data Folder
Find real data here [Soap and Kiwi](https://drive.google.com/drive/folders/1JrTtno7c-FnYuNJ044FKbjlZYujJiczN?usp=sharing).




