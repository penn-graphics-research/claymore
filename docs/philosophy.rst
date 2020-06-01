Design Philosophy
=================

Data-Oriented Design
--------------------
Due to the increased overhead of memory over compute operations, 
**data-oriented design** philosophy has been widely adopted in software design. 
It focuses on the data layout that is efficient for certain patterns of memory access. 
Usually, the optimization is achieved through better utilization of caches.

Here are some of the most popular libraries providing efficient data structures.

- `OpenVDB <https://github.com/AcademySoftwareFoundation/openvdb>`_: OpenVDB is an open source C++ library comprising a novel hierarchical data structure and a large suite of tools for the efficient storage and manipulation of sparse volumetric data discretized on three-dimensional grids.
- `SPGrid <https://orionquest.github.io/papers/SSPGASS/paper.html>`_: SPGrid is a new data structure for compact storage and efficient stream processing of sparsely populated uniform Cartesian grids. It leverages the extensive hardware acceleration mechanisms inherent in the x86 Virtual Memory Management system to deliver sequential and stencil access bandwidth comparable to dense uniform grids.
- `Cabana <https://github.com/ECP-copa/Cabana/wiki/AoSoA>`_: As a generalization of SoA (Struct-of-Arrays) and AoS (Array-of-Structs) layouts, AoSoA (Array-of-Structs-of-Arrays), which is adopted by **Cabana**, appears to be a robust choice for both `CPU <https://github.com/ECP-copa/Cabana/wiki/Benchmarks>`_ and `GPU <https://www.seas.upenn.edu/~cffjiang/research/wang2020multigpu/wang2020multigpu.pdf>`_

Zero-Overhead Principle 
-----------------------
By C++ committee, the zero-overhead principle states that: What you don’t use, you don’t pay for (in time or space) and further: What you do use, you couldn’t hand code any better. 
As Bjarne Stroustrup points out, it requires programming techniques, language features, and implementations to achieve such a zero-overhead abstraction.
In C++ language, designing an almost zero-overhead abstraction is sophisticated and difficult. 
But it is worth working towards this goal. 
Meanwhile, such abstractions must provide more benefit than cost.

Despite all the differences in the interpretations of *cost* (See `Chandler Carruth's talk <https://www.youtube.com/watch?v=rHIkrotSwcc&t=821s>`_), it is of great importance to reduce both the runtime cost as well as the build time cost (especially for C++). 
Sometimes there even has to be a trade-off.
So we adopt the following practices in our codebase:

- Use constexpr if feasible
- Avoid recursive template instantiations
- Encapsulate reusable units into precompiled libraries
- ...
