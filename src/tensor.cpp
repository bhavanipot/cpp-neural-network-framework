// A tensor generalizes scalars (0-D), vectors (1-D), and matrices (2-D) to any
// number of dimensions. In a neural network, inputs, weights, activations, and
// gradients are all tensors, e.g. a batch of images has shape {N, C, H, W}.

#include "nnf/tensor.h"
#include <sstream> //for to_string implementation
#include <stdexcept> //for error messages
#include <random> //for randn implementation

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

//create a tensor from with chosen numbers
Tensor Tensor::from_vector(const std::vector<float>& data, const Shape& shape) {

    //throw error if the numbers don't fit the shape
    if (data.size() != shape_numel(shape))
        throw std::invalid_argument("from_vector: data size doesn't match shape");

    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<std::vector<float>>(data);  //copy your numbers in
    impl->shape = shape;
    impl->strides = contiguous_strides(shape);
    return Tensor(impl);
}

// Create a tensor filled with random numbers.
Tensor Tensor::randn(const Shape& shape, unsigned seed) {
    std::mt19937 generator(seed);  //set up the random number generator,    
    std::normal_distribution<float> bell_curve(0.0f, 1.0f); // a bell curve with center 0 and spread 1

    Tensor result = zeros(shape); // make a tensor of the right size (all zeros for now)

    // replace every number with a random one
    std::vector<float>& numbers = *result.impl_->storage;
    for (size_t i = 0; i < numbers.size(); ++i) {
        numbers[i] = bell_curve(generator);
    }

    return result;
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

std::string Tensor::to_string() const {
    std::ostringstream out;

    // Part 1: the shape
    out << "Tensor(shape={";
    for (size_t i = 0; i < impl_->shape.size(); ++i) {
        if (i > 0) out << ", ";
        out << impl_->shape[i];
    }

    // Part 2: the numbers
    out << "}, data=[";
    const auto& data = *impl_->storage;
    for (size_t i = 0; i < numel(); ++i) {
        if (i > 0) out << ", ";
        out << data[impl_->offset + i];
    }
    out << "])";

    return out.str();
}

// add two tensors elementwise
//[1,2,3] + [4,5,6] = [5,7,9]
Tensor operator+(const Tensor& a, const Tensor& b) {

    //both tensors must be the same size
    if (a.shape() != b.shape())
        throw std::invalid_argument("shape mismatch");

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    const float* b_nums = b.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] + b_nums[i];
    }

    return result;
}

// subtract two tensors elementwise
//[1,2,3] - [4,5,6] = [-3,-3,-3]
Tensor operator-(const Tensor& a, const Tensor& b) {

    //both tensors must be the same size
    if (a.shape() != b.shape())
        throw std::invalid_argument("shape mismatch");

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    const float* b_nums = b.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] - b_nums[i];
    }

    return result;
}

// multiply two tensors elementwise
//[1,2,3] * [4,5,6] = [4,10,18]
Tensor operator*(const Tensor& a, const Tensor& b) {

    //both tensors must be the same size
    if (a.shape() != b.shape())
        throw std::invalid_argument("shape mismatch");

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    const float* b_nums = b.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] * b_nums[i];
    }

    return result;
}

// divide two tensors elementwise
//[1,2,3] / [4,5,6] = [0.25,0.4,0.5]
Tensor operator/(const Tensor& a, const Tensor& b) {

    //both tensors must be the same size
    if (a.shape() != b.shape())
        throw std::invalid_argument("shape mismatch");

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    const float* b_nums = b.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] / b_nums[i];
    }

    return result;
}

}  // namespace nnf


