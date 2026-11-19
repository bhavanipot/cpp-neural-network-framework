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
- [x] Autograd
- [x] Training (layers, optimizers, MNIST)
- [ ] Fast CPU kernels (cache blocking, threads, SIMD)

## Results

A two-layer fully connected network (784 → 128 → 10) reaches **96.1% test accuracy** on MNIST
after three epochs of training. With cache-friendly loop ordering and multithreaded matrix
multiplication, one epoch over the 60,000 training images takes **~0.7 s** on an Apple M2 CPU,
down from ~14 s with the naive implementation (a ~20× speed-up).

## Next steps

**Improving MNIST accuracy**
- [ ] Longer training, wider networks, and the Adam optimizer
- [ ] Convolutional layers that exploit the spatial structure of images, targeting 99%+

**Real-time heart rhythm monitoring**

The main goal ahead is a real-time arrhythmia detector: a model that reads an
electrocardiogram (ECG) as it streams in and flags abnormal heartbeats, running entirely on
the CPU with this framework.

- [ ] **Data:** load the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
      (48 half-hour, two-channel ECG recordings with ~110,000 beats annotated by cardiologists),
      reading its WFDB signal and annotation files directly
- [ ] **Preprocessing:** filter baseline wander and noise, and cut the signal into windows
      centred on each heartbeat
- [ ] **Model:** 1-D convolutional layers that learn the shape of a heartbeat over time,
      classifying beats into the standard AAMI groups (normal, supraventricular, ventricular,
      fusion, unknown)
- [ ] **Honest evaluation:** an inter-patient split, so the model is tested on people it has
      never seen, and per-class sensitivity and positive predictive value  
- [ ] **Real-time demo:** stream a recording at its true speed (360 samples per second), flag
      abnormal beats as they occur, and measure the time taken to classify each beat