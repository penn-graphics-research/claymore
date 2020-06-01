Overview
=============

This is the opensource code **Claymore** for the SIGGRAPH 2020 paper:

**A Massively Parallel and Scalable Multi-GPU Material Point Method** 

`page <https://sites.google.com/view/siggraph2020-multigpu>`_, 
`pdf <https://www.seas.upenn.edu/~cffjiang/research/wang2020multigpu/wang2020multigpu.pdf>`_, 
`supp <https://www.seas.upenn.edu/~cffjiang/research/wang2020multigpu/supp.pdf>`_, 
`video <https://vimeo.com/414136257>`_

Authors: 
`Xinlei Wang\* <https://github.com/littlemine>`_,
`Yuxing Qiu\* <https://yuxingqiu.github.io/>`_,
`Stuart R. Slattery <https://www.ornl.gov/staff-profile/stuart-r-slattery>`_, 
`Yu Fang <http://squarefk.com/>`_, 
`Minchen Li <https://www.seas.upenn.edu/~minchenl/>`_, 
`Song-Chun Zhu <http://www.stat.ucla.edu/~sczhu/>`_, 
`Yixin Zhu <https://yzhu.io/>`_, 
`Min Tang <https://min-tang.github.io/home/>`_, 
`Dinesh Manocha <https://www.cs.umd.edu/people/dmanocha>`_,
`Chenfanfu Jiang <https://www.seas.upenn.edu/~cffjiang/>`_
(* Equal contributions)

Introduction
------------

Harnessing the power of modern multi-GPU architectures, we present a massively parallel simulation system based on the Material Point Method (MPM) for simulating physical behaviors of materials undergoing complex topological changes, self-collision, and large deformations. 
Our system makes three critical contributions:

- Introduce a new particle data structure that promotes coalesced memory access patterns on the GPU and eliminates the need for complex atomic operations on the memory hierarchy when writing particle data to the grid. 
- Propose a kernel fusion approach using a new Grid-to-Particles-to-Grid (G2P2G) scheme, which efficiently reduces GPU kernel launches, improves latency, and significantly reduces the amount of global memory needed to store particle data. 
- Introduce optimized algorithmic designs that allow for efficient sparse grids in a shared memory context, enabling us to best utilize modern multi-GPU computational platforms for hybrid Lagrangian-Eulerian computational patterns. 

We demonstrate the effectiveness of our method with extensive benchmarks, evaluations, and dynamic simulations with elastoplasticity, granular media, and fluid dynamics. In comparisons against an open-source and heavily optimized CPU-based MPM codebase on an elastic sphere colliding scene with particle counts ranging from 5 to 40 million, our GPU MPM achieves over 100X per-time-step speedup on a workstation with an Intel 8086K CPU and a single Quadro P6000 GPU, exposing exciting possibilities for future MPM simulations in computer graphics and computational science. 

Moreover, compared to the state-of-the-art GPU MPM method, we not only achieve 2X acceleration on a single GPU but our kernel fusion strategy and Array-of-Structs-of-Array (AoSoA) data structure design also generalizes to multi-GPU systems. 
Our multi-GPU MPM exhibits near-perfect weak and strong scaling with 4 GPUs, enabling performant and large-scale simulations on a 1024x1024x1024 grid with close to 100 million particles with less than 4 minutes per frame on a single 4-GPU workstation and 134 million particles with less than 1 minute per frame on an 8-GPU workstation.

Gallery
------------

.. image:: images/examples.jpg

Bibtex
------

Please cite our paper if you use this code for your research: 

.. bibliography:: paper.bib
   :list: bullet
   :all:

.. code-block:: none

    @article{Wang2020multiGMPM, 
        author = {Xinlei Wang* and Yuxing Qiu* and Stuart R. Slattery and Yu Fang and Minchen Li and Song-Chun Zhu and Yixin Zhu and Min Tang and Dinesh Manocha and Chenfanfu Jiang},
        title = {A Massively Parallel and Scalable Multi-GPU Material Point Method},
        journal = {ACM Transactions on Graphics},
        year = {2020},
        volume = {39},
        number = {4},
        articleno = {Article 30}
    }
