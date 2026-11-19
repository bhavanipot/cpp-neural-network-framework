// Time matmul at the size MNIST training uses, to see how fast each speed-up makes it.
// Build in Release first: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

#include <chrono>
#include <iostream>
#include "nnf/tensor.h"

//milliseconds since start, using C++'s stopwatch
static double ms_since(std::chrono::steady_clock::time_point start) {
    std::chrono::duration<double, std::milli> passed = std::chrono::steady_clock::now() - start;
    return passed.count();
}

int main() {
    //one batch of 64 pictures through the first MNIST layer: {64, 784} x {784, 128}
    size_t n = 64;
    size_t m = 784;
    size_t p = 128;
    nnf::Tensor x = nnf::Tensor::randn({n, m}, 1);
    nnf::Tensor W = nnf::Tensor::randn({m, p}, 2);
    int repeats = 100;  //time many runs and take the average, one run is too quick to measure well

    //forward only
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        nnf::matmul(x, W);
    }
    double forward_ms = ms_since(start) / repeats;

    //forward and backward, like a training step
    W.set_requires_grad(true);
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        W.zero_grad();
        nnf::sum(nnf::matmul(x, W)).backward();
    }
    double both_ms = ms_since(start) / repeats;

    //GFLOP/s: billions of multiplies and adds per second
    //a matmul does n × m × p multiplies and the same number of adds
    double operations = 2.0 * n * m * p;
    std::cout << "matmul {" << n << "," << m << "} x {" << m << "," << p << "}\n";
    std::cout << "forward:            " << forward_ms << " ms, "
              << operations / (forward_ms * 1e6) << " GFLOP/s\n";
    std::cout << "forward + backward: " << both_ms << " ms\n";
}
