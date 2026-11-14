#pragma once
#include <vector>
#include "nnf/tensor.h"

namespace nnf {

// Stochastic gradient descent: nudges every weight against its blame.
//usage each training step: zero_grad(), then loss.backward(), then step()
class SGD {
public:
    SGD(const std::vector<Tensor>& params, float learning_rate); //the weights to look after
    void zero_grad(); //clear the blame on every weight
    void step();      //nudge every weight: w -= learning_rate × blame

private:
    std::vector<Tensor> params_;  //the weights it looks after
    float learning_rate_;         //how big each nudge is
};

}  // namespace nnf
