#pragma once
#include <memory>
#include <vector>
#include <cstddef>
#include <utility>
#include <string>

namespace nnf {

using Shape = std::vector<size_t>;

//helper functions 
size_t shape_numel(const Shape& shape); //count the numbers
std::vector<size_t> contiguous_strides(const Shape& shape); //work out the stride sizes

struct TensorImpl {
    std::shared_ptr<std::vector<float>> storage;  // all numbers, in one flat list
    Shape shape;                                  // e.g. {2, 3}
    std::vector<size_t> strides;                  // e.g. {3, 1}
    size_t offset = 0;                            // where this tensor starts in storage
};

//Tensor class
class Tensor {
public:
    static Tensor full(const Shape& shape, float value); //making tensors
    static Tensor zeros(const Shape& shape) { return full(shape, 0.0f); }
    static Tensor from_vector(const std::vector<float>& data, const Shape& shape); //tensor from your own numbers
    static Tensor randn(const Shape& shape, unsigned seed = 42); //tensor of random numbers

    const Shape& shape() const { return impl_->shape; } //size dimensions 
    size_t numel() const { return shape_numel(impl_->shape); } //numbers it can hold
    std::string to_string() const; //printing the tensors for debugging

    float& at(const std::vector<size_t>& index); //finding value at index

private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<TensorImpl> impl_;
};

}  // namespace nnf
