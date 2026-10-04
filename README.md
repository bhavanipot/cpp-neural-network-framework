# cpp-neural network framework

A deep learning framework built from scratch in C++20.

_🚧 Work in progress._

## What is this?

This repository is a small framework for training neural networks on a CPU, with no external
libraries. It is under active development, and will include tensors, automatic differentiation, layers, optimizers, and fast math kernels.

## Why?

High-level libraries make training a model a few lines of code, which also hides how
it actually works. This project is a way to understand deep learning from the ground up, and
to explore how fast training can be on ordinary, CPU-only hardware.

## Roadmap

- [x] Tensors
- [ ] Autograd
- [ ] Training (layers, optimizers, MNIST)
- [ ] Fast CPU kernels (cache blocking, threads, SIMD)
