// Optimizers: they hold the list of weights and nudge them all after each backward().

#include "nnf/optim.h"
#include <stdexcept>

namespace nnf {

// remember the weights and the learning rate
SGD::SGD(const std::vector<Tensor>& params, float learning_rate)
    : params_(params), learning_rate_(learning_rate) {

    //every weight needs a blame box, or step() would have nothing to read
    for (const Tensor& p : params_) {
        if (!p.requires_grad())
            throw std::invalid_argument("SGD: every weight must be tracked (set_requires_grad(true))");
    }
}

// clear the blame on every weight, ready for the next step
void SGD::zero_grad() {
    for (Tensor& p : params_) {
        p.zero_grad();
    }
}

// nudge every weight against its blame: w -= learning_rate × blame
//changes the numbers in place, so no receipt is written
void SGD::step() {
    for (Tensor& p : params_) {
        Tensor blame = p.grad();

        //look up where the numbers live once, before the loop: asking p.data() inside the
        //loop repeats the lookup every time and stops the compiler using SIMD (about 12x slower)
        float* weights = p.data();
        const float* blame_nums = blame.data();
        size_t count = p.numel();

        for (size_t i = 0; i < count; ++i) {
            weights[i] -= learning_rate_ * blame_nums[i];
        }
    }
}

}  // namespace nnf
