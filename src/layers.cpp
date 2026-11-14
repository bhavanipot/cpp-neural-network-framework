// Layers: building blocks of a network, each holding its own weights.

#include "nnf/layers.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace nnf {

// make W random (scaled so outputs stay about the size of inputs) and b zero, both tracked
//the scale √(2 ÷ inputs) is called He initialization, and suits relu between layers
Linear::Linear(size_t in_features, size_t out_features, unsigned seed)
    : in_features_(in_features),
      W_(Tensor::randn({in_features, out_features}, seed) * std::sqrt(2.0f / static_cast<float>(in_features))),
      b_(Tensor::zeros({out_features})) {

    W_.set_requires_grad(true);
    b_.set_requires_grad(true);
}

// out = x · W + b, with b repeated for every example in the batch
Tensor Linear::forward(const Tensor& x) const {

    //x must be {batch, in_features}
    if (x.shape().size() != 2 || x.shape()[1] != in_features_)
        throw std::invalid_argument("Linear: input must be {batch, " + std::to_string(in_features_) + "}");

    return matmul(x, W_) + b_;
}

// the weights the optimizer should look after
std::vector<Tensor> Linear::parameters() const {
    return {W_, b_};
}

}  // namespace nnf
