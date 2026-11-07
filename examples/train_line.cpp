// Teach the framework the line y = 2x + 1 from examples, using autograd.
// Each step: guess, measure how wrong (loss), backward() for the blame, then nudge w and b against their blame.

#include <iostream>
#include "nnf/tensor.h"

//nudge every number against its blame: t -= learning_rate × blame
//changes t's numbers in place, so no receipt is written
static void nudge(nnf::Tensor& t, float learning_rate) {
    nnf::Tensor blame = t.grad();
    for (size_t i = 0; i < t.numel(); ++i) {
        t.data()[i] -= learning_rate * blame.data()[i];
    }
}

int main() {
    //the examples: inputs x and the right answers y = 2x + 1
    nnf::Tensor x = nnf::Tensor::from_vector({-2, -1, 0, 1, 2, 3}, {6});
    nnf::Tensor y = x * 2.0f + 1.0f;

    //what we want to learn, starting from a bad guess
    nnf::Tensor w = nnf::Tensor::zeros({1});
    nnf::Tensor b = nnf::Tensor::zeros({1});
    w.set_requires_grad(true);
    b.set_requires_grad(true);

    const float learning_rate = 0.1f;  //how big each nudge is

    for (int step = 0; step <= 100; ++step) {
        //1. clear last step's blame
        w.zero_grad();
        b.zero_grad();

        //2. guess, and measure how wrong the guesses are
        nnf::Tensor pred = x * w + b;
        nnf::Tensor diff = pred - y;
        nnf::Tensor loss = nnf::mean(diff * diff);

        //3. work out the blame
        loss.backward();

        //4. nudge against the blame
        nudge(w, learning_rate);
        nudge(b, learning_rate);

        if (step % 10 == 0)
            std::cout << "step " << step << ": loss " << loss.sum()
                      << ", w " << w.data()[0] << ", b " << b.data()[0] << "\n";
    }
}
