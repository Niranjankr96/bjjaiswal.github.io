---
title: High Performance Computing in Python
category: Parallel-Computing
tags: Parallel-Computing
excerpt: |
    Python as an interpreted language has been considered too slow for high-performance computing. However, recent development of CUDA based libraries in python has changed the programming paradigms, capable of rapid iterative development of Python with harnessing the power of NVIDIA GPUs. 
feature_image: "https://d2ufo47lrtsv5s.cloudfront.net/content/sci/358/6370/1530/F1.large.jpg?width=800&height=600&carousel=1"
image: "https://d2ufo47lrtsv5s.cloudfront.net/content/sci/358/6370/1530/F1.large.jpg?width=800&height=600&carousel=1"

---
Python, however started as scripting language, has advanced its primarily purpose. Now, Python has become general-purpose programming language not just for scientific and numerical computing but helps in providing mechanisms and technique to link and interpret with compiled code like c/c++, Java, Assembly, etc,. This linking and interpreting machine level code of compiled code has provided capability of enhancing computational performance like level of C/C++ performance. The Open Source Software Contribution and enormous user and Developer community have created a rich ecosystem of third party libraries for numerical and scientific computing, data analysis, machine learning, high performance computing, Big Data, and many more.

### Why Python's Performance is equivalent to any compiled code like C/C++ program.
It is true that Python is slower than C/C++ or any machine or low-level programming language. However, Python's tendency of linking and interpreting compiled code has superseded its limitation of time computation. To make Python work faster, Developer makes software with some functionality written in Python and other core number-crunching parts of software written in C or Fortran and compile in such a manner as to be callable from Python.

> API : Application Programming Interface, the interface used by software components to communicate with each other. For example, when a main program calls a subprogram, the calling sequence for the subprogram is the API.

Many times such linking is done by Python-C API, when the Python interpreter calls a compiled library (written in C or some other compiled language which can be called from C).

{% include figure.html image="/assets/img/Python-hpc/Python-C-API.png" caption="Python-C API" %}

The figure shows two scenario where the same computation is performed but through different set of Function calls. Imagine, we are looping thought a container with 10 elements and performing some computation on those elements within the compiled library. The left panel shows how repeated calls from interpreter to compiled library can increase the time computation whereas right panel shows how a bundled set of compiled computations triggered by a single call from the interpreter can improve the run time performance. **Lesson 1** : `This should be kept in mind while implementing HPC's programs or software in python`.

### How Python scientific computing ecosystem helps in HPC?
Libraries of Python scientific computing ecosystem provides greater numerical efficiencies and implementation of variety of useful ecosystem. These libraries are also capable of holding large amounts of data without exhausting RAM.

{% include figure.html image="http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1509622333/scipy-eco_kqi2su.png" caption="Scientific Ecosystem, Credit: Datacamp.org" %}

Numpy is Python's core numerical library. Numpy is implemented in such a way that its underlying has C representation or better say implemented in C or Fortran underneath. Numpy makes use of lower-level libraries such as **[BLAS](http://www.netlib.org/blas/)**, **[LAPACK](http://www.netlib.org/lapack/)** and **[ATLAS](http://math-atlas.sourceforge.net/)**

<!--
> BLAS: Basic Linear Algebra Subroutines

> LAPACK: Linear ALgebra PACKage

>ATLAS: Automatically Tuned Linear Algebra Software, provides C and Fortran77 interfaces to portably efficient BLAS implementation, as well as a few routines from LAPACK.

### Pythonic CUDA
#### [PyCUDA](https://documen.tician.de/pycuda/)
#### [Numba](https://numba.pydata.org/)
#### [Pyculib](http://pyculib.readthedocs.io/en/latest/#)
[](https://www.anaconda.com/blog/developer-blog/open-sourcing-anaconda-accelerate/)
-->
