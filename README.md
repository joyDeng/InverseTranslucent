# psdr-cuda
This is a path space differentiable renderer (psdr-cuda) with bssrdf support which forked from [psdr-cuda](https://psdr-cuda.readthedocs.io/en/latest/). The derivation and experimental detail can be found in [paper](https://www.cs.cornell.edu/~xideng/pub/deng22dsss.pdf).


## Compile
To build the code, please follow this process of [compiling psdr-cuda](https://psdr-cuda.readthedocs.io/en/latest/core_compile.html)

## Materials
The material we use in reconstructing is `HeterSub`, the definition and implementation can be found in `hetersub.h` and `hetersub.cpp`, currently, it is using dipole model, one can switch to better diphole in the future. The expression of diphole model and better diphole model can be found in [this report](http://www.eugenedeon.com/wp-content/uploads/2014/04/betterdipole.pdf).

## Integrators

Here are several integrators implemented in this framework, the one we used for reconstruction is the `DirectIntegrator`, it is a direct light integrator with point light assumption, support gradient estimation of geometry with bssrdf material (both primary and secondary discontintuity). The `OldDirectIntegrator` correspond to the `DirectIntegrator` in the original psdr-cuda codebase; and the `ColocateIntegrator` is integrator with the "point light locates at the same place as camera" assumpition. To validate the implementation of BSSRDF we used the `LaserIntegrator`, where a directional coherent light is pointing down to a plane, it is deprecated now.


## Optimizing
#### Synthetic data 
To do: adding sythetic examples ... 

#### Real data 
1. The latest optimizing code is in `"examples/python/learn_real_data.py"`.

2. To set the `ROOT_DIR`, `SCENE_DIR`, etc... go to `"examples/python/constants.py"`.

3. To optimize homogeneous value of `alebdo` and `sigma_t`, set `args.sigma_texture = 0`.

4. To optimize `albedo` and `sigma_t` as texture, set `args.sigma_texture = 512` and `args.albedo_texture = 512`.


## Data Folder
Find real data here [Soap and Kiwi](https://drive.google.com/drive/folders/1JrTtno7c-FnYuNJ044FKbjlZYujJiczN?usp=sharing).




