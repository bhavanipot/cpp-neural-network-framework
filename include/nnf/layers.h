#pragma once
#include <vector>
#include "nnf/tensor.h"

namespace nnf {

// A fully connected layer: out = x · W + b
//x has shape {batch, in}, out has shape {batch, out}
class Linear {
public:
    Linear(size_t in_features, size_t out_features, unsigned seed = 42);

    Tensor forward(const Tensor& x) const;    //matmul(x, W) + b
    std::vector<Tensor> parameters() const;   //{W, b}, to hand to the optimizer

    Tensor weight() const { return W_; }      //W, shape {in, out}
    Tensor bias() const { return b_; }        //b, shape {out}

private:
    size_t in_features_;  //how many numbers go in per example
    Tensor W_;            //the weights
    Tensor b_;            //the bias
};

}  // namespace nnf
