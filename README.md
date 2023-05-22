# HIP Programming Examples

This repository contains examples used in the *High-Performance GPU Programming with HIP Course*.

## Table of Content


### Section 1: Introduction

#### Example 1: Example C++ Program

**Path**: [sec_01/ex_01](sec_01/ex_01)

**Video**: Section 1, Video 3

**Description**: This is an example of HelloWorld program written in C++. This program is not intended to be executed, but only to demonstrate the instructions that CPU cores run. 

### Section 2: Parallel Programming

### Example 1: pthread Creation and Synchronization

**Path**: [sec_02/ex_01](sec_02/ex_01)

**Video**: Section 2, Video 2

**Description**: This is an simple example of pthread programming.

### Example 2: Passing Arguments to pthreads

**Path**: [sec_02/ex_02](sec_02/ex_02)

**Video**: Section 2, Video 3

**Description**: This example demonstrates how to pass arguments to pthreads.


### Example 3: Sychronizing pthreads

**Path**: [sec_02/ex_03](sec_02/ex_03)

**Video**: Section 2, Video 4

**Description**: This example demonstrates how to use mutex and atomics to avoid data races. You can change the pthread creation function to switch the method.

### Example 4: Calculating PI with pthreads

**Path**: [sec_02/ex_04](sec_02/ex_04)

**Video**: Section 2, Video 5

**Description**: This example demonstrates how to calculate PI in parallel. 

### Example 5: C++ Threads

**Path**: [sec_02/ex_05](sec_02/ex_05)

**Video**: Section 2, Video 6

**Description**: This example demonstrates how to use C++ threads.

### Example 6: MPI Threads

**Path**: [sec_02/ex_06](sec_02/ex_06)

**Video**: Section 2, Video 7

**Description**: This example demonstrates how to use MPI threads.



### Section 3: HIP Programming 1

### Example 1: HIP Hello World

**Path**: [sec_03/ex_01](sec_03/ex_01)

**Video**: Section 3, Video 1

**Description**: This example demonstrates an example of Hello World program on a HIP-compatible GPU.

### Example 2: Vector Addition

**Path**: [sec_03/ex_02](sec_03/ex_02)

**Video**: Section 3, Video 4

**Description**: This example demonstrates an example of Vector Addition program in HIP. 

### Example 3: Matrix Multiplication

**Path**: [sec_03/ex_03](sec_03/ex_03)

**Video**: Section 3, Video 5

### Section 4: HIP Programming 2

### Example 1: Struct

**Path**: [sec_04/ex_01](sec_04/ex_01/)

**Video**: Section 4, Video 3

**Description**: This example demonstrates the calculation of distance to the origin in HIP using the concept of a struct. 

### Example 2: HIP Stream

**Path**: [sec_04/ex_02](sec_04/ex_02)

**Video**: Section 4, Video 4

**Description**: This example demonstrates how to use two separate kernels and two streams to compute the square and cube of an input array. 

### Example 3: Pinned Memory

**Path**: [sec_04/ex_03](sec_04/ex_03)

**Video**: Section 4, Video 5

**Description**: This example demonstrates the utilization of pinned memory in HIP by modifying the previous vector addition example. 

### Example 4: Unified Memory

**Path**: [sec_04/ex_04](sec_04/ex_04)

**Video**: Section 4, Video 5

**Description**: This example demonstrates the utilization of unified memory in HIP by modifyinf the previous vector addition example.

### Example 5: HIP Events

**Path**: [sec_04/ex_05](sec_04/ex_05)

**Video**: Section 4, Video 6

**Description**: This example demonstrates the utilization of HIP events to accurately measure the execution time of a vector addition program. 

### Section 6: ROCm Tools

### Example 1: Application Tracing

**Path**: [sec_06/ex_01](sec_06/ex_01)

**Video**: Section 6, Video 5

**Description**: This example demonstrates all the files generated after performing application tracing with the ROCm Profiler on a vector addition program. 

### Example 2: GPU Profiling

**Path**: [sec_06/ex_02](sec_06/ex_02)

**Video**: Section 6, Video 5

**Description**: This example demonstrates the file generated after performing GPU profiling on a vection addition program by specifying an input file. 

### Example 3: Hipify-perl

**Path**: [sec_06/ex_03](sec_06/ex_03)

**Video**: Section 6, Video 6

**Description**: This example shows the conversion of CUDA code to HIP code using the Hipify-perl tool, resulting in the demonstration of the converted HIP code. 

### Example 4: Hipify-clang

**Path**: [sec_06/ex_04](sec_06/ex_04)

**Video**: Section 6, Video 6

**Description**: This example shows the conversion of CUDA code to HIP code using the Hipify-clang tool, resulting in the demonstration of the converted HIP code. 


## Section 7: HIP Performance Improvement 1

### Example 1: Image Gamma Correction (Simple)

**Path**: [sec_07/ex_01](sec_07/ex_01)

**Video**: Section 7, Video 2

**Description**: This example demonstrates how to use HIP to implement an image
gamma correction algorithm.

### Example 2: Image Gamma Correction (Fixed-Sized Kernel)

**Path**: [sec_07/ex_02](sec_07/ex_02)

**Video**: Section 7, Video 2

**Description**: This example demonstrates how to use HIP to implement an image
gamma correction algorithm. This example uses a fixed-sized kernel to reduce 
the overhead of block dispatching.