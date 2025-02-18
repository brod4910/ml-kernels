# ML Kernels
A simple tensor library implemented in C++ designed for iterative optimization of various ML kernels.

<p align="center">
  <img src="https://i.imgur.com/NzP2zWat.png"/>
</p>

# Simple Frontend
All algorithms that have been implemented can be accessed through the `mlkl::` namespace and all algorithms have various verions denoted with `vX` that can be accesed within the `mlkl::operators::cuda::` namespace. For example, `mlkl::operators::cuda::sgemm_v4`.

The tensor class is barebones and is meant to be created using the creation functions such as `mlkl::empty` or `mlkl::randn`.

A simple tensor allocator class (`TensorAllocator`) is provided for managing data when testing various functions. Tensors are not managed otherwise since they don't have smart-pointer functionality. If you create a tensor not using the `TensorAllocator` then you need to manually use `mlkl::destory`.

# Examples
Every kernel has an example that times and measures the total GFLOPs processed. Each kernel is tested on various input shapes for a certain amount of iterations. Some of the examples may have references that utilize CUDA functions such as CUBLAS for the `sgemm-example`.

# Design Flaws
Designing a tensor library has many design choices and I have encountered several in this simple implementation.

- How to handle allocation/deallocation 
    - Allocate a pointer and need to dealloc for memory leaks
    - Internal counter similar to shared_ptr for destruction
    - memory arena
- Handling different Dtypes 
    - dynamic casting 
    - Pimpl implementations of various Dtypes
    - templates


Many of the choices come to a head when you want performance, usability, or small implementation size. For this simple library, I mostly chose usability for the tensor portion of this library since I'm mostly concerned with implementing the various kernels. However, choosing one trade-off sometimes pigeon-holes the entire design of the library such as the choice to not use internal `shared_ptr` semantics for the Tensor class for destruction. The choice to not use `shared_ptr` semantics means we need to pass around raw pointers and leaves destruction up to the user.
