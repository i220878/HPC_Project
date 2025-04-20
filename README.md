# HPC Project (WIP)
Image Recognition Neural Network Parallelization for High Performance Computing with GPU's Course. 

The purpose of this project is to analyze multiple implemented solutions of the MNIST Classification problem, which is a fundamental machine learning task that involves recognizing handwritten digits (0-9) from a dataset of 70,000 grayscale images (28x28 pixels each). It serves as a benchmark for evaluating machine learning models. 

In this specific project, the same native (V1) algorithm will be implemented multiple times. The goal is to observe how much of a speedup can be obtained by parallelization of the task using GPU Programming. The implementations are as follows:

- Native (V1): Single Core CPU
- GPU (V2): Parallelization via GPU Programming in CUDA
- GPU (V3): Optimized version of V2 which uses Launch Configuration, Occupancy, Communication Optimizations, and Memory Optimizations with focus on Memory Hierarchy
- GPU (V4): Tensor Core version of V3

## Steps to compile and analyze V1

Go to directory and run these commands:
```bash
make
gprof nn.exe gmon.out > analysis.txt
```

## Steps to compile and run V2, V3, V4

```bash
nvcc -o run <filename>.cu
./run
```

## Steps to compile and run V5

```bash
nvc -acc -o run V5.c
./run
```
