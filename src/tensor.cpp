// A tensor generalizes scalars (0-D), vectors (1-D), and matrices (2-D) to any
// number of dimensions. In a neural network, inputs, weights, activations, and
// gradients are all tensors, e.g. a batch of images has shape {N, C, H, W}.

#include "nnf/tensor.h"
#include <sstream> //for to_string implementation
#include <stdexcept> //for error messages
#include <random> //for randn implementation
#include <cmath> //for exp, log, tanh
#include <algorithm> //for std::max

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

//for ordered data
static void require_contiguous(const Tensor& t) {
    if (!t.is_contiguous())
        throw std::invalid_argument("tensor must be contiguous (e.g. not a transpose)");
}

//the shape two tensors combine to, e.g. {2,3} and {3} -> {2,3}
//sizes are compared from the right: they must be equal, or one must be 1 (or missing)
static Shape broadcast_shape(const Shape& a, const Shape& b) {
    size_t ndim = std::max(a.size(), b.size());
    Shape result(ndim);

    //line the shapes up from the right and compare each pair of sizes
    for (size_t i = 0; i < ndim; ++i) {
        size_t a_size = i < a.size() ? a[a.size() - 1 - i] : 1;  //missing counts as 1
        size_t b_size = i < b.size() ? b[b.size() - 1 - i] : 1;

        if (a_size != b_size && a_size != 1 && b_size != 1)
            throw std::invalid_argument("shapes can't be broadcast together");

        result[ndim - 1 - i] = std::max(a_size, b_size);
    }

    return result;
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

// same data, different shape
//for example, {6} -> {2,3}
Tensor Tensor::reshape(const Shape& new_shape) const {

    //the new shape must hold the same number of values as the old one
    if (shape_numel(new_shape) != numel())
        throw std::invalid_argument("reshape: element count must stay the same");

    //must be in order
    require_contiguous(*this);

    auto impl = std::make_shared<TensorImpl>();
    impl->storage = impl_->storage;  //share the same numbers
    impl->shape = new_shape;
    impl->strides = contiguous_strides(new_shape);
    impl->offset = impl_->offset;

    return Tensor(impl);
}

// swap rows and columns without copying; only the shape and strides are swapped.
//for example {2,3} -> {3,2}
Tensor Tensor::transpose() const {

    //transpose only works on 2-D tensors (matrices)
    if (impl_->shape.size() != 2)
        throw std::invalid_argument("transpose: tensor must be 2-D");

    auto impl = std::make_shared<TensorImpl>();
    impl->storage = impl_->storage;  //share the same numbers
    impl->shape = {impl_->shape[1], impl_->shape[0]};
    impl->strides = {impl_->strides[1], impl_->strides[0]};
    impl->offset = impl_->offset;

    return Tensor(impl);
}

// repeat this tensor along new dimensions
//{3} [10,20,30] -> {2,3} strides {0,1}, reads as [10,20,30, 10,20,30]
Tensor Tensor::expand(const Shape& new_shape) const {

    //the new shape must be one this tensor can be repeated into
    if (broadcast_shape(impl_->shape, new_shape) != new_shape)
        throw std::invalid_argument("expand: can't repeat into that shape");

    size_t extra = new_shape.size() - impl_->shape.size();  //new dimensions added on the left

    auto impl = std::make_shared<TensorImpl>();
    impl->storage = impl_->storage;  //share the same numbers
    impl->shape = new_shape;
    impl->strides.assign(new_shape.size(), 0);  //stride 0 = repeat
    impl->offset = impl_->offset;

    //dimensions that already had the right size keep their normal stride
    for (size_t d = 0; d < impl_->shape.size(); ++d) {
        if (impl_->shape[d] == new_shape[extra + d])
            impl->strides[extra + d] = impl_->strides[d];
    }

    return Tensor(impl);
}

// check if the tensor is contiguous in memory,(if the strides are the normal ones for this shape)
//{2,3} with strides {3,1} -> true
//transpose {3,2} with strides {1,3} -> false
bool Tensor::is_contiguous() const {
    return impl_->strides == contiguous_strides(impl_->shape);
}

// make a contiguous copy of the tensor in memory, with standard strides for its shape
//a transpose {3,2} with storage [1,2,3,4,5,6] -> storage [1,4,2,5,3,6]
Tensor Tensor::contiguous() const {

    //already in order
    if (is_contiguous())
        return *this;

    Tensor result = zeros(impl_->shape);

    const auto& data = *impl_->storage;
    float* result_nums = result.data();
    std::vector<size_t> index(impl_->shape.size(), 0);  //start at {0, 0, ...}

    for (size_t i = 0; i < numel(); ++i) {

        //turn the index into a position in storage 
        size_t pos = impl_->offset;
        for (size_t d = 0; d < index.size(); ++d) {
            pos += index[d] * impl_->strides[d];
        }
        result_nums[i] = data[pos];

        //carry the index
        for (size_t d = index.size(); d-- > 0;) {
            if (++index[d] < impl_->shape[d]) break;
            index[d] = 0;
        }
    }

    return result;
}

std::string Tensor::to_string() const {
    std::ostringstream out;

    // Part 1: the shape
    out << "Tensor(shape={";
    for (size_t i = 0; i < impl_->shape.size(); ++i) {
        if (i > 0) out << ", ";
        out << impl_->shape[i];
    }

    // Part 2: the numbers (an in-order copy, so views print correctly)
    out << "}, data=[";
    Tensor ordered = contiguous();
    const float* nums = ordered.data();
    for (size_t i = 0; i < numel(); ++i) {
        if (i > 0) out << ", ";
        out << nums[i];
    }
    out << "])";

    return out.str();
}

// add up every number into one total
//[1,2,3] -> 6
float Tensor::sum() const {
    require_contiguous(*this);

    const float* nums = data();

    float total = 0.0f;
    for (size_t i = 0; i < numel(); ++i) {
        total += nums[i];
    }

    return total;
}

// the average of every number
//[1,2,3] -> 2  (6 / 3)
float Tensor::mean() const {
    //an empty tensor has no average
    if (numel() == 0)
        throw std::invalid_argument("mean: tensor is empty");

    return sum() / static_cast<float>(numel());
}

// add up along one axis of a 2-D tensor
//[[1,2,3],[4,5,6]]: axis 0 (down the columns) -> [5,7,9]
Tensor Tensor::sum(size_t axis) const {

    //only 2-D tensors (matrices) for now, with axis 0 or 1
    if (impl_->shape.size() != 2 || axis > 1)
        throw std::invalid_argument("sum: needs a 2-D tensor and axis 0 or 1");

    require_contiguous(*this);

    size_t rows = impl_->shape[0];
    size_t cols = impl_->shape[1];

    //axis 0 leaves one total per column, axis 1 one total per row
    Tensor result = zeros({axis == 0 ? cols : rows});

    const float* nums = data();
    float* result_nums = result.data();

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            size_t target = (axis == 0) ? col : row;
            result_nums[target] += nums[row * cols + col];
        }
    }

    return result;
}

// the average along one axis of a 2-D tensor
//[[1,2,3],[4,5,6]]: axis 1 -> [2,5]
Tensor Tensor::mean(size_t axis) const {

    //sum checks the tensor and axis first, so the shape lookup below is safe
    Tensor total = sum(axis);

    //an empty axis has no average
    if (impl_->shape[axis] == 0)
        throw std::invalid_argument("mean: axis is empty");

    return total / static_cast<float>(impl_->shape[axis]);
}

// add two tensors elementwise
//[1,2,3] + [4,5,6] = [5,7,9]
Tensor operator+(const Tensor& a, const Tensor& b) {

    //stretch both to a shared shape, e.g. {2,3} and {3} -> both {2,3}
    Shape shape = broadcast_shape(a.shape(), b.shape());
    Tensor a_full = a.expand(shape).contiguous();
    Tensor b_full = b.expand(shape).contiguous();

    Tensor result = Tensor::zeros(shape);

    const float* a_nums = a_full.data();
    const float* b_nums = b_full.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] + b_nums[i];
    }

    return result;
}

// subtract two tensors elementwise
//[1,2,3] - [4,5,6] = [-3,-3,-3]
Tensor operator-(const Tensor& a, const Tensor& b) {

    //stretch both to a shared shape, e.g. {2,3} and {3} -> both {2,3}
    Shape shape = broadcast_shape(a.shape(), b.shape());
    Tensor a_full = a.expand(shape).contiguous();
    Tensor b_full = b.expand(shape).contiguous();

    Tensor result = Tensor::zeros(shape);

    const float* a_nums = a_full.data();
    const float* b_nums = b_full.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] - b_nums[i];
    }

    return result;
}

// multiply two tensors elementwise
//[1,2,3] * [4,5,6] = [4,10,18]
Tensor operator*(const Tensor& a, const Tensor& b) {

    //stretch both to a shared shape, e.g. {2,3} and {3} -> both {2,3}
    Shape shape = broadcast_shape(a.shape(), b.shape());
    Tensor a_full = a.expand(shape).contiguous();
    Tensor b_full = b.expand(shape).contiguous();

    Tensor result = Tensor::zeros(shape);

    const float* a_nums = a_full.data();
    const float* b_nums = b_full.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] * b_nums[i];
    }

    return result;
}

// divide two tensors elementwise
//[1,2,3] / [4,5,6] = [0.25,0.4,0.5]
Tensor operator/(const Tensor& a, const Tensor& b) {

    //stretch both to a shared shape, e.g. {2,3} and {3} -> both {2,3}
    Shape shape = broadcast_shape(a.shape(), b.shape());
    Tensor a_full = a.expand(shape).contiguous();
    Tensor b_full = b.expand(shape).contiguous();

    Tensor result = Tensor::zeros(shape);

    const float* a_nums = a_full.data();
    const float* b_nums = b_full.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] / b_nums[i];
    }

    return result;
}

// flip the sign of every number
//-[1,-2,3] = [-1,2,-3]
Tensor operator-(const Tensor& a) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = -a_nums[i];
    }

    return result;
}

// keep every number the same, return a new tensor with the same values
//+[1,-2,3] = [1,-2,3]
Tensor operator+(const Tensor& a) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i];
    }

    return result;
}

// add a number to every entry
//[1,2,3] + 10 = [11,12,13]
Tensor operator+(const Tensor& a, float b) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] + b;
    }

    return result;
}

// subtract a number from every entry
//[1,2,3] - 10 = [-9,-8,-7]
Tensor operator-(const Tensor& a, float b) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] - b;
    }

    return result;
}

// multiply every entry by a number
//[1,2,3] * 10 = [10,20,30]
Tensor operator*(const Tensor& a, float b) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] * b;
    }

    return result;
}

// divide every entry by a number
//[1,2,3] / 10 = [0.1,0.2,0.3]
Tensor operator/(const Tensor& a, float b) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] / b;
    }

    return result;
}

// same as above, but with the number written on the left
//10 + [1,2,3] = [11,12,13]
Tensor operator+(float a, const Tensor& b) {
    return b + a;
}

//10 - [1,2,3] = [9,8,7]
Tensor operator-(float a, const Tensor& b) {

    require_contiguous(b);

    Tensor result = Tensor::zeros(b.shape());

    const float* b_nums = b.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a - b_nums[i];
    }

    return result;
}

//10 * [1,2,3] = [10,20,30]
Tensor operator*(float a, const Tensor& b) {
    return b * a;
}

//10 / [1,2,3] = [10,5,3.33]
Tensor operator/(float a, const Tensor& b) {

    require_contiguous(b);

    Tensor result = Tensor::zeros(b.shape());

    const float* b_nums = b.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a / b_nums[i];
    }

    return result;
}

// matrix multiply: {n, m} x {m, p} -> {n, p}
Tensor matmul(const Tensor& a, const Tensor& b) {

    //matmul only works on 2-D tensors (matrices)
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw std::invalid_argument("matmul: both tensors must be 2-D");

    size_t n = a.shape()[0];  //rows of a, rows of the result
    size_t m = a.shape()[1];  //columns of a, must match rows of b
    size_t p = b.shape()[1];  //columns of b, columns of the result

    //the inner dimensions must match, e.g. {2,3} x {3,4} -> {2,4}
    if (m != b.shape()[0])
        throw std::invalid_argument("matmul: inner dimensions must match");

    //both tensors must be stored in order
    require_contiguous(a);
    require_contiguous(b);

    Tensor result = Tensor::zeros({n, p});

    const float* a_nums = a.data();
    const float* b_nums = b.data();
    float* result_nums = result.data();

    //each result cell is a row of a dotted with a column of b
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < p; ++col) {
            float total = 0.0f;
            for (size_t k = 0; k < m; ++k) {
                total += a_nums[row * m + k] * b_nums[k * p + col];
            }
            result_nums[row * p + col] = total;
        }
    }

    return result;
}

// set every negative number to 0
//relu([-2,0,3]) = [0,0,3]
Tensor relu(const Tensor& a) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = a_nums[i] > 0.0f ? a_nums[i] : 0.0f;
    }

    return result;
}

// e to the power of every number
//exp([0,1,2]) = [1,2.718,7.389]
Tensor exp(const Tensor& a) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = std::exp(a_nums[i]);
    }

    return result;
}

// natural log of every number, log(0) gives -inf and log of a negative gives nan
//log([1,2.718]) = [0,1]
Tensor log(const Tensor& a) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = std::log(a_nums[i]);
    }

    return result;
}

// every number into the range (-1, 1)
//tanh([-10,0,10]) = [-1,0,1]
Tensor tanh(const Tensor& a) {

    require_contiguous(a);

    Tensor result = Tensor::zeros(a.shape());

    const float* a_nums = a.data();
    float* result_nums = result.data();

    for (size_t i = 0; i < result.numel(); ++i) {
        result_nums[i] = std::tanh(a_nums[i]);
    }

    return result;
}

}  // namespace nnf


