System
======

To conveniently reuse the code required for a specific type of tasks, we build a precompiled and customized library, i.e. **system**.
It is essentially a **Singleton** class which can be selectively included and utilized by projects.

CUDA
----

**CUDA** system provides CUDA related utilities, including 
- setting up the GPU devices, constructing necessary resources and enabling certain features.
- temporary memory pools of various memory type for intermediate computations.
- context handles, each corresponding to one GPU, through which programmers launch kernels, synchronize among streams, manage memory, etc.

Also, a few helper functions that can be called on the host side or the device side are also included.

IO
----

**IO** system currently only provide what is necessary for the MPM project, e.g. reading geometry data and outputting generated particle data asynchronously.