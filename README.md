# psdr-cuda
Path-space differentiable renderer with bssrdf support.

## Integrator
DirectIntegrator:

DirectIntegrator with point light assumption, support gradient estimation of geometry with bssrdf material (both primary and secondary discontintuity).

ColocateIntegrator:
This is integrator with the "point light locates at the same place as camera" assumpition.

LaserIntegrator: 
This is only used in bssrdf validation, where a directional coherent light is pointing down to the plane.

OldDirectIntegrator: 
This is the origin direct integrator in psdr-cuda, which support bsdf with MIS direct lighting.

## Optimizing both geometry and texture

The latest optimizing code is in "examples/python/learn_real_data.py"

To set the ROOT_DIR, SCENE_DIR, etc... go to "examples/python/constants.py"

To optimize homogeneous value of alebdo and sigma_t, use args.sigma_texture = 0

To optimize albedo and sigma_t as texture, use args.sigma_texture = 512 and args.albedo_texture = 512: 


### Data Folder
Find real data here [Soap and Kiwi](https://drive.google.com/drive/folders/1JrTtno7c-FnYuNJ044FKbjlZYujJiczN?usp=sharing).




