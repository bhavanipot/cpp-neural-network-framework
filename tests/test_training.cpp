// Test: train y = 2x + 1 from examples
// Run with: cmake --build build && ctest --test-dir build --output-on-failure

#include "test_helpers.h"  //fail, check_number, check_tensor, report

// t -= learning_rate × blame
//changes t's numbers in place
static void nudge(Tensor& t, float learning_rate) {
    Tensor blame = t.grad();
    for (size_t i = 0; i < t.numel(); ++i) {
        t.data()[i] -= learning_rate * blame.data()[i];
    }
}

static void test_learns_a_line() {
    //the examples: inputs x and the right answers y = 2x + 1
    Tensor x = Tensor::from_vector({-2, -1, 0, 1, 2, 3}, {6});
    Tensor y = x * 2.0f + 1.0f;

    //start from a far guess
    Tensor w = Tensor::zeros({1});
    Tensor b = Tensor::zeros({1});
    w.set_requires_grad(true);
    b.set_requires_grad(true);

    for (int step = 0; step < 200; ++step) {
        w.zero_grad();
        b.zero_grad();

        Tensor diff = x * w + b - y;
        nnf::mean(diff * diff).backward();

        nudge(w, 0.1f);
        nudge(b, 0.1f);
    }

    //after training, w and b should be very close to the real line
    check_number("learned w", w.data()[0], 2.0f, 0.01f);
    check_number("learned b", b.data()[0], 1.0f, 0.01f);
}

int main() {
    test_learns_a_line();
    return report();
}
