# A Massively Parallel and Scalable Multi-GPU Material Point Method

<div align="left">
    <a href="https://claymore.readthedocs.io/en/latest/"> Documentation </a>
</div>

## Description

This is the opensource code for the SIGGRAPH 2020 paper:

**A Massively Parallel and Scalable Multi-GPU Material Point Method** 

[page](https://sites.google.com/view/siggraph2020-multigpu)\, [pdf](https://www.seas.upenn.edu/~cffjiang/research/wang2020multigpu/wang2020multigpu.pdf)\, [supp](https://www.seas.upenn.edu/~cffjiang/research/wang2020multigpu/supp.pdf)\, [video](https://vimeo.com/414136257)

Authors:
[Xinlei Wang](https://github.com/littlemine)\*, 
[Yuxing Qiu](https://yuxingqiu.github.io/)\*, 
[Stuart R. Slattery](https://www.ornl.gov/staff-profile/stuart-r-slattery), 
[Yu Fang](http://squarefk.com/), 
[Minchen Li](https://www.seas.upenn.edu/~minchenl/), 
[Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/), 
[Yixin Zhu](https://yzhu.io/), 
[Min Tang](https://min-tang.github.io/home/), 
[Dinesh Manocha](https://www.cs.umd.edu/people/dmanocha)
[Chenfanfu Jiang](https://www.seas.upenn.edu/~cffjiang/)
(* Equal contributions)

<p float="left">
<img src="Clips/examples.jpg" />
</p>

Harnessing the power of modern multi-GPU architectures, we present a massively parallel simulation system based on the Material Point Method (MPM) for simulating physical behaviors of materials undergoing complex topological changes, self-collision, and large deformations. Our system makes three critical contributions. First, we introduce a new particle data structure that promotes coalesced memory access patterns on the GPU and eliminates the need for complex atomic operations on the memory hierarchy when writing particle data to the grid. Second, we propose a kernel fusion approach using a new Grid-to-Particles-to-Grid (G2P2G) scheme, which efficiently reduces GPU kernel launches, improves latency, and significantly reduces the amount of global memory needed to store particle data. Finally, we introduce optimized algorithmic designs that allow for efficient sparse grids in a shared memory context, enabling us to best utilize modern multi-GPU computational platforms for hybrid Lagrangian-Eulerian computational patterns. We demonstrate the effectiveness of our method with extensive benchmarks, evaluations, and dynamic simulations with elastoplasticity, granular media, and fluid dynamics. In comparisons against an open-source and heavily optimized CPU-based MPM codebase on an elastic sphere colliding scene with particle counts ranging from 5 to 40 million, our GPU MPM achieves over 100X per-time-step speedup on a workstation with an Intel 8086K CPU and a single Quadro P6000 GPU, exposing exciting possibilities for future MPM simulations in computer graphics and computational science. Moreover, compared to the state-of-the-art GPU MPM method, we not only achieve 2X acceleration on a single GPU but our kernel fusion strategy and Array-of-Structs-of-Array (AoSoA) data structure design also generalizes to multi-GPU systems. Our multi-GPU MPM exhibits near-perfect weak and strong scaling with 4 GPUs, enabling performant and large-scale simulations on a 1024x1024x1024 grid with close to 100 million particles with less than 4 minutes per frame on a single 4-GPU workstation and 134 million particles with less than 1 minute per frame on an 8-GPU workstation.


<!--
<p float="left">
<img src="Data/Clips/faceless.gif" height="128px"/>
<img src="Data/Clips/flow.gif" height="128px"/>
<img src="Data/Clips/chains.gif" height="128px"/>
<img src="Data/Clips/cat.gif" height="128px"/>
</p>
-->

## Compilation
This is a cross-platform C++/CUDA cmake project. The minimum version requirement of cmake is 3.15, yet the latest version is generally recommended. The required CUDA version is 10.2.

Currently, *supported OS* includes Windows 10 and Ubuntu (>=18.04), and *tested compilers* includes gcc8.4, msvc v142, clang-9 (includes msvc version). 

### Build
Run the following command in the *root directory*.
```mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Or configure the project using the *CMake Tools* extension in *Visual Studio Code* (recommended).


### Data

Currently, binary position data and the level-set (signed distance field) data are accepted as input files for particles. Uniformly sampling particles from analytic geometries is another viable way for the initialization of models.

### Run Demos
The project provides the following GPU-based schemes for MPM:
- improved single-GPU pipeline
- static geometry (particle) partitioning multi-GPU pipeline
<!--
- dynamic spatial partitioning multi-GPU pipeline
-->

Go to *Projects/\*\**, run the executable.

## Code Usage
> Use the codebase in another cmake c++ project.

Directly include the codebase as a submodule, and follow the examples in the *Projects*.

> Develop upon the codebase.

Create a sub-folder in *Projects* with a cmake file at its root.

## Bibtex

Please cite our paper if you use this code for your research: 
```
@article{Wang2020multiGMPM,
    author = {Xinlei Wang* and Yuxing Qiu* and Stuart R. Slattery and Yu Fang and Minchen Li and Song-Chun Zhu and Yixin Zhu and Min Tang and Dinesh Manocha and Chenfanfu Jiang},
    title = {A Massively Parallel and Scalable Multi-GPU Material Point Method},
    journal = {ACM Transactions on Graphics},
    year = {2020},
    volume = {39},
    number = {4},
    articleno = {Article 30}
}
```

## Credits
This project draws inspirations from [Taichi](https://github.com/taichi-dev/taichi), [GMPM](https://github.com/kuiwuchn/GPUMPM).

### Acknowledgement
We thank Yuanming Hu for useful discussions and proofreading, Feng Gao for his help on configuring workstations. We appreciate Prof. Chenfanfu Jiang and Yuanming Hu for their insightful advice on the documentation.

### Dependencies
The following libraries are adopted in our project development:

- [cub](http://nvlabs.github.io/cub/)
- [fmt](https://fmt.dev/latest/index.html)

For particle data IO and generation, we use these two libraries in addition:

- [partio](http://partio.us/)
- [SDFGen](https://github.com/christopherbatty/SDFGen)

Due to the C++ standard requirement (at most C++14) for compiling CUDA (10.2) code, we import these following libraries as well:

- [function_ref](https://github.com/TartanLlama/function_ref)
- [optional](https://github.com/TartanLlama/optional)
- [variant](https://github.com/mpark/variant)