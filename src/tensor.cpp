// A tensor generalizes scalars (0-D), vectors (1-D), and matrices (2-D) to any
// number of dimensions. In a neural network, inputs, weights, activations, and
// gradients are all tensors, e.g. a batch of images has shape {N, C, H, W}.

#include "nnf/tensor.h"
#include <stdexcept> //for error messages

namespace nnf {

//count the amount of data my tensor holds
size_t shape_numel(const Shape& shape) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

//find the stride lengths
std::vector<size_t> contiguous_strides(const Shape& shape) {
    std::vector<size_t> strides(shape.size());
    size_t step = 1;
    for (size_t d = shape.size(); d-- > 0;) {
        strides[d] = step;
        step *= shape[d];
    }
    return strides;
}

//create a tensor
Tensor Tensor::full(const Shape& shape, float value) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<std::vector<float>>(shape_numel(shape), value);
    impl->shape = shape;
    impl->strides = contiguous_strides(shape);
    return Tensor(impl);
}

float& Tensor::at(const std::vector<size_t>& index) {

    //throw error if out of index
    if (index.size() != impl_->shape.size())
        throw std::out_of_range("wrong number of indices");

    //find the positon
    size_t pos = impl_->offset;
    for (size_t d = 0; d < index.size(); ++d) {
        if (index[d] >= impl_->shape[d])
            throw std::out_of_range("index out of range");
        pos += index[d] * impl_->strides[d];
    }

    //return the value at that position
    return (*impl_->storage)[pos];
}

}  // namespace nnf


