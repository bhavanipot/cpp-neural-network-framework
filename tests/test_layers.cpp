// Tests for layers.
// Run with: cmake --build build && ctest --test-dir build --output-on-failure

#include <cmath>
#include <stdexcept>
#include "test_helpers.h"  //fail, check_number, check_tensor, report
#include "nnf/layers.h"
#include "nnf/optim.h"

static void test_linear_shapes() {
    nnf::Linear layer(3, 2);

    check_number("W shape rows", layer.weight().shape()[0], 3.0f);
    check_number("W shape cols", layer.weight().shape()[1], 2.0f);
    check_tensor("b starts at zero", layer.bias(), {2}, {0, 0});
    check_number("W is tracked", layer.weight().requires_grad(), 1.0f);
    check_number("b is tracked", layer.bias().requires_grad(), 1.0f);
    check_number("parameters are W and b", layer.parameters().size(), 2.0f);

    Tensor batch = Tensor::zeros({4, 3});  //4 examples, 3 features each
    Tensor out = layer.forward(batch);
    check_number("output rows", out.shape()[0], 4.0f);
    check_number("output cols", out.shape()[1], 2.0f);

    try {
        layer.forward(Tensor::zeros({4, 5}));
        fail("Linear with the wrong number of inputs should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_linear_math() {
    //write known numbers into W and b, then check forward gives x · W + b
    nnf::Linear layer(2, 2);
    Tensor W = layer.weight();  //a handle to the layer's real W, not a copy
    Tensor b = layer.bias();
    float w_nums[] = {1, 2, 3, 4};  //W = [[1, 2], [3, 4]]
    for (size_t i = 0; i < 4; ++i) W.data()[i] = w_nums[i];
    b.data()[0] = 10;
    b.data()[1] = 20;

    Tensor x = Tensor::from_vector({1, 1, 2, 0}, {2, 2});  //two examples: [1,1] and [2,0]
    //[1,1]·W = [4,6] + b = [14,26];  [2,0]·W = [2,4] + b = [12,24]
    check_tensor("linear forward", layer.forward(x), {2, 2}, {14, 26, 12, 24});
}

static void test_linear_init() {
    //the typical weight size should be close to √(2 ÷ inputs)
    nnf::Linear big(1000, 100);
    Tensor W = big.weight();
    float typical = std::sqrt((W * W).mean());
    check_number("weights are scaled for 1000 inputs", typical, std::sqrt(2.0f / 1000.0f), 0.005f);

    //different seeds give different starting weights
    nnf::Linear first(3, 2, 1);
    nnf::Linear second(3, 2, 2);
    if (first.weight().data()[0] == second.weight().data()[0])
        fail("different seeds should give different weights");
}

static void test_linear_learns_a_line() {
    //the y = 2x + 1 example again, now through a Linear(1, 1) layer
    Tensor x = Tensor::from_vector({-2, -1, 0, 1, 2, 3}, {6, 1});
    Tensor y = x * 2.0f + 1.0f;

    nnf::Linear layer(1, 1);
    nnf::SGD opt(layer.parameters(), 0.1f);

    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();
        Tensor diff = layer.forward(x) - y;
        nnf::mean(diff * diff).backward();
        opt.step();
    }

    check_number("layer learned w", layer.weight().data()[0], 2.0f, 0.01f);
    check_number("layer learned b", layer.bias().data()[0], 1.0f, 0.01f);
}

int main() {
    test_linear_shapes();
    test_linear_math();
    test_linear_init();
    test_linear_learns_a_line();
    return report();
}
