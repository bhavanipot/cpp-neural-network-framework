// Tests for saving and loading weights.
// Run with: cmake --build build && ctest --test-dir build --output-on-failure

#include <stdexcept>
#include <string>
#include "test_helpers.h"  //fail, check_number, check_tensor, report
#include "nnf/layers.h"
#include "nnf/save.h"

static void test_save_and_load() {
    //two layers with different random weights
    nnf::Linear first(3, 2, 1);
    nnf::Linear second(3, 2, 2);
    Tensor b = first.bias();
    b.data()[0] = 0.5f;  //so the bias isn't just zeros
    b.data()[1] = -1.25f;

    //copy first's weights into second through a file
    nnf::save_parameters(first.parameters(), "test_weights.txt");
    nnf::load_parameters(second.parameters(), "test_weights.txt");

    //now both should give exactly the same outputs (tolerance 0)
    Tensor x = Tensor::from_vector({1, 2, 3, -4, 5, 0.25f}, {2, 3});
    Tensor out_first = first.forward(x);
    Tensor out_second = second.forward(x);
    for (size_t i = 0; i < 4; ++i) {
        check_number("loaded output " + std::to_string(i), out_second.data()[i], out_first.data()[i], 0.0f);
    }

    //loading writes into the same tensors, so they stay tracked and ready to train
    check_number("loaded W still tracked", second.weight().requires_grad(), 1.0f);
}

static void test_bad_files() {
    //weights of a Linear(3, 2) don't fit a Linear(2, 2)
    nnf::Linear three_inputs(3, 2);
    nnf::Linear two_inputs(2, 2);
    nnf::save_parameters(three_inputs.parameters(), "test_weights.txt");
    try {
        nnf::load_parameters(two_inputs.parameters(), "test_weights.txt");
        fail("loading weights of the wrong shape should throw");
    } catch (const std::runtime_error&) {}

    //a file that doesn't exist
    try {
        nnf::load_parameters(two_inputs.parameters(), "no-such-file.txt");
        fail("loading a missing file should throw");
    } catch (const std::runtime_error&) {}
}

int main() {
    test_save_and_load();
    test_bad_files();
    return report();
}
