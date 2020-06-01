Compilation
=============

This is a cross-platform C++/CUDA cmake project. 
The minimum version requirement of **CMake** is 3.15, although the latest version is generally recommended. 
The recommended version of **CUDA** is 10.2.

Currently tested C++ compilers (as the host compiler for **NVCC**) on different platforms include:

  +----------+------------------+
  | platform | Compilers        |
  +==========+==================+
  | Windows  | msvc142, clang-9 |
  +----------+------------------+
  | Linux    | gcc8.4, clang-9  |
  +----------+------------------+

In short, the supported compilers should support **C++14** standard and be in compliance with **NVCC**.
Since the future releases of **CUDA** are officially excluded on Mac OS, and there is a more suitable candidate **Metal** developed by **Apple**, the **Mac OS** platform is not discussed.

External Dependencies
---------------------
These libraries are very helpful in developing this project and save a lot of efforts:

- `CUB <http://nvlabs.github.io/cub/>`_ provides state-of-the-art, reusable software components for every layer of the CUDA programming model including many parallel primitives and utilities.

- `fmt <https://fmt.dev/latest/index.html>`_ is an open-source formatting library for C++. 

These libraries are used for particle data IO and initialization:

- `partio <http://partio.us/>`_ is an open source C++ library for reading, writing and manipulating a variety of standard particle formats (GEO, BGEO, PTC, PDB, PDA).

- `SDFGen <https://github.com/christopherbatty/SDFGen>`_ is a simple commandline utility to generate grid-based signed distance field (level set) from triangle meshes, using code from Robert Bridson's website.

Build Commands
-------------------
Run the following command in the *root directory*.

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release

The project can also be configured through other interfaces, e.g. using the *CMake Tools* extension in *Visual Studio Code* (recommended), in *Visual Studio Studio*, or in *CMake GUI*.
