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