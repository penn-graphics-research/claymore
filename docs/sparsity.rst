Sparse Data Structures
======================

In many graphics applications, e.g. ray tracing, collision detection, neighborhood search, etc., there exists a large amount of queries upon spatial data structures, 
and the involved volumetric data is often spatially sparse. Dedicated data structures like `OpenVDB <https://github.com/AcademySoftwareFoundation/openvdb>`_ 
and `SPGrid <https://orionquest.github.io/papers/SSPGASS/paper.html>`_ are introduced to provide efficient access to sparse data for specific operations.
However, there is no best universal data structure for different types of sparsity and access patterns. 

Through the previous **Hierarchy Composition** interface, we can compose and define sparse data structures very easily 
in order to experiment and identify an appropriate candidate for our need. 
In our experiences, the sparse data structures can be categorized in terms of their underlying memory resource.

- **Utilizing Virtual Memory**: rely on the virtual memory support from the underlying OS, driver and hardware.
- **Manually Managing Memory**: manually maintain the sparse data, sometimes including a mapping from the discrete identifiers (e.g. spatial coordinates) to a contiguous sequence of indices.

Then, providing allocators of specific memory resources to the **Structural Instance**, we complete defining the variable (instance), 
the internal structure of which is specified by the **Structural Node**. Here we illustrate two common key aspects.

Allocation
----------

Utilizing Virtual Memory
````````````````````````
The automatic management of the virtual memory can help relieve the burden of manual maintenance. By avoiding frequent page-faults, 
the access to the virtual memory can be as efficient as the heap or global memory (CUDA).

Depending on the location of the memory we want to allocate for the instance, the allocation APIs are different.

Allocation On Host
''''''''''''''''''
- Windows

.. code-block:: cpp

    void* ptr = VirtualAlloc(nullptr, totalBytes, MEM_RESERVE, PAGE_READWRITE);

- Linux

.. code-block:: cpp

    void* ptr = mmap(nullptr, totalBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);

Allocation On Device (CUDA)
'''''''''''''''''''''''''''
.. code-block:: cpp

    void* ptr;
    cudaMallocManaged((void **)&ptr, totalBytes);

Manually Managing Memory
````````````````````````
The above method heavily relies on the support from the driver, and the granularity of the data unit should be on the same level as a page in the virtual memory system.
Therefore it is often a more robust choice to manually manage the memory.

Allocation On Host
''''''''''''''''''
.. code-block:: cpp

    void* ptr = (void *)malloc(totalBytes);

Allocation On Device (CUDA)
'''''''''''''''''''''''''''
.. code-block:: cpp

    void* ptr;
    cudaMalloc((void **)&ptr, totalBytes);

Modeling Sparsity
-----------------
The underlying memory resource intended for the sparse data structure largely influences the design of the data structure itself.
Here we discuss multiple definitions of the *SPGrid-variants* using different strategies.

Utilizing Virtual Memory
````````````````````````
The virtual memory system can save the efforts of modeling the sparsity information. Simply defining the grid like a dense grid is sufficient.

.. code-block:: cpp

    // domain
    using BlockDomain = domain<char, 4, 4, 4>;
    using GridDomain = domain<int, g_grid_size, g_grid_size, g_grid_size>;
    // decorator
    using DefaultDecorator = decorator<structural_allocation_policy::full_allocation, structural_padding_policy::compact, attrib_layout::soa>;
    // structural node
    using grid_block_ = structural<structural_type::dense, DefaultDecorator, BlockDomain, f32_, f32_, f32_, f32_>;
    using grid_ = structural<structural_type::dense, DefaultDecorator, GridDomain, grid_block_>;


Manually Managing Memory
````````````````````````
When adopting this strategy, the programmer should additionally maintain the mapping from spatial block coordinates to a contiguous sequence of indices.
Usually we store this mapping through a (spatial) hash table or a lookup table. 
And the resulting data structure for the *SPGrid-variant* essentially becomes an array of grid blocks.

.. code-block:: cpp

    using GridBufferDomain = domain<int, g_max_active_block>;
    using grid_buffer_ = structural<structural_type::dynamic, DefaultDecorator, GridBufferDomain, grid_block_>;

