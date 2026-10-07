#pragma once
#include <memory>
#include <vector>
#include <cstddef>
#include <utility>

namespace nnf {

using Shape = std::vector<size_t>;

size_t shape_numel(const Shape& shape);
std::vector<size_t> contiguous_strides(const Shape& shape);

struct TensorImpl {
    std::shared_ptr<std::vector<float>> storage;  // all numbers, in one flat list
    Shape shape;                                  // e.g. {2, 3}
    std::vector<size_t> strides;                  // e.g. {3, 1}
    size_t offset = 0;                            // where this tensor starts in storage
};

class Tensor {
public:
    static Tensor full(const Shape& shape, float value);
    static Tensor zeros(const Shape& shape) { return full(shape, 0.0f); }

    const Shape& shape() const { return impl_->shape; }
    size_t numel() const { return shape_numel(impl_->shape); }

    float& at(const std::vector<size_t>& index);

private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<TensorImpl> impl_;
};

}  // namespace nnf
