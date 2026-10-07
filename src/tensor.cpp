#include "nnf/tensor.h"
#include <stdexcept>

namespace nnf {

size_t shape_numel(const Shape& shape) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

std::vector<size_t> contiguous_strides(const Shape& shape) {
    std::vector<size_t> strides(shape.size());
    size_t step = 1;
    for (size_t d = shape.size(); d-- > 0;) {
        strides[d] = step;
        step *= shape[d];
    }
    return strides;
}

Tensor Tensor::full(const Shape& shape, float value) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<std::vector<float>>(shape_numel(shape), value);
    impl->shape = shape;
    impl->strides = contiguous_strides(shape);
    return Tensor(impl);
}

float& Tensor::at(const std::vector<size_t>& index) {
    // 1. start with: size_t pos = impl_->offset;
    // 2. loop over each dimension d, adding index[d] * impl_->strides[d] to pos
    // 3. return (*impl_->storage)[pos];
}

}  // namespace nnf

if (index.size() != impl_->shape.size())
    throw std::out_of_range("wrong number of indices");
