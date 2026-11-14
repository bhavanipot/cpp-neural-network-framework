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
        for (size_t i = 0; i < p.numel(); ++i) {
            p.data()[i] -= learning_rate_ * blame.data()[i];
        }
    }
}

}  // namespace nnf
