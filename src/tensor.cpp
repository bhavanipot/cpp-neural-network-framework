// A tensor generalizes scalars (0-D), vectors (1-D), and matrices (2-D) to any
// number of dimensions. In a neural network, inputs, weights, activations, and
// gradients are all tensors, e.g. a batch of images has shape {N, C, H, W}.

#include "nnf/tensor.h"
#include <sstream> //for to_string implementation
#include <stdexcept> //for error messages
#include <random> //for randn implementation
#include <cmath> //for exp, log, tanh
#include <algorithm> //for std::max
#include <set> //for backward's visited list

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

//t's numbers in reading order, even if t is a view like a transpose
//walks every index like an odometer and reads each number through the strides
static std::vector<float> values_of(const TensorImpl& t) {
    size_t count = shape_numel(t.shape);
    std::vector<float> nums(count);

    const auto& data = *t.storage;
    std::vector<size_t> index(t.shape.size(), 0);  //start at {0, 0, ...}

    for (size_t i = 0; i < count; ++i) {

        //turn the index into a position in storage
        size_t pos = t.offset;
        for (size_t d = 0; d < index.size(); ++d) {
            pos += index[d] * t.strides[d];
        }
        nums[i] = data[pos];

        //carry the index
        for (size_t d = index.size(); d-- > 0;) {
            if (++index[d] < t.shape[d]) break;
            index[d] = 0;
        }
    }

    return nums;
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

//which number of a (possibly broadcast) input made position i of the result
//e.g. result {2,3}, input {3}: positions 0..5 came from 0,1,2,0,1,2
//     result {2,3}, input {2,1}: positions 0..5 came from 0,0,0,1,1,1
static size_t source_position(size_t i, const Shape& result_shape, const Shape& input_shape) {
    size_t pos = 0;
    size_t step = 1;  //how far one step along this dimension moves in the input

    //peel off one index per dimension, from the last, like reading the digits of a number
    for (size_t d = result_shape.size(); d-- > 0;) {
        size_t index = i % result_shape[d];
        i /= result_shape[d];

        //line the input's dimensions up from the right; missing or size-1 ones were repeated
        size_t from_right = result_shape.size() - 1 - d;
        if (from_right < input_shape.size()) {
            size_t input_size = input_shape[input_shape.size() - 1 - from_right];
            if (input_size != 1)
                pos += index * step;
            step *= input_size;
        }
    }

    return pos;
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

    Tensor result(impl);

    //if this tensor is tracked, write the receipt line for autograd
    if (impl_->requires_grad)
        result.record(Op::Reshape, {*this});

    return result;
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

    Tensor result(impl);

    //if this tensor is tracked, write the receipt line for autograd
    if (impl_->requires_grad)
        result.record(Op::Transpose, {*this});

    return result;
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

    //copy the numbers out in reading order
    Tensor result = from_vector(values_of(*impl_), impl_->shape);

    //if this tensor is tracked, write the receipt line for autograd
    if (impl_->requires_grad)
        result.record(Op::Contiguous, {*this});

    return result;
}

// start or stop tracking this tensor for autograd
//tracking gives it a grad of all zeros, ready for backward() to fill in
void Tensor::set_requires_grad(bool on) {
    impl_->requires_grad = on;

    if (on)
        impl_->grad.assign(numel(), 0.0f);  //one blame number per value, all 0 to start
    else
        impl_->grad.clear();                //not tracked, so no blame to store
}

// the blame for each number, as a new tensor of the same shape
//[1,2,3] tracked -> grad starts as [0,0,0]
Tensor Tensor::grad() const {

    //only tracked tensors have a grad
    if (!impl_->requires_grad)
        throw std::invalid_argument("grad: tensor isn't tracked, call set_requires_grad(true) first");

    return from_vector(impl_->grad, impl_->shape);
}

// the name of an operation, for printing
std::string op_name(Op op) {
    if (op == Op::Add) return "Add";
    if (op == Op::Mul) return "Mul";
    if (op == Op::Sum) return "Sum";
    if (op == Op::Mean) return "Mean";
    if (op == Op::Sub) return "Sub";
    if (op == Op::Div) return "Div";
    if (op == Op::AddScalar) return "AddScalar";
    if (op == Op::MulScalar) return "MulScalar";
    if (op == Op::ScalarSub) return "ScalarSub";
    if (op == Op::ScalarDiv) return "ScalarDiv";
    if (op == Op::Relu) return "Relu";
    if (op == Op::Exp) return "Exp";
    if (op == Op::Log) return "Log";
    if (op == Op::Tanh) return "Tanh";
    if (op == Op::Matmul) return "Matmul";
    if (op == Op::Transpose) return "Transpose";
    if (op == Op::Reshape) return "Reshape";
    if (op == Op::Contiguous) return "Contiguous";
    if (op == Op::SumAxis) return "SumAxis";
    return "None";
}

// write this tensor's receipt line: which operation made it, and from which inputs
void Tensor::record(Op op, const std::vector<Tensor>& inputs, float scalar) {
    impl_->op = op;
    impl_->scalar = scalar;

    impl_->parents.clear();
    for (const Tensor& input : inputs) {
        impl_->parents.push_back(input.impl_);
    }

    //the result is tracked too, so blame can pass through it on the way back
    set_requires_grad(true);
}

// one of the inputs that made this tensor, e.g. parent(0) of a * b is a
Tensor Tensor::parent(size_t i) const {

    //i must be one of the parents
    if (i >= impl_->parents.size())
        throw std::out_of_range("parent: no parent at that position");

    return Tensor(impl_->parents[i]);
}

//add t to the list after every tensor it was made from (inputs before results)
static void build_order(const std::shared_ptr<TensorImpl>& t,
                        std::vector<std::shared_ptr<TensorImpl>>& order,
                        std::set<TensorImpl*>& visited) {

    //a tensor used twice (like p in p * q + p) must only be listed once
    if (visited.count(t.get()))
        return;
    visited.insert(t.get());

    //list the inputs first, then this tensor
    for (const auto& parent : t->parents) {
        build_order(parent, order, visited);
    }
    order.push_back(t);
}

//a + b: nudging a or b by 1 nudges the result by 1, so both get the blame unchanged;
//if one was broadcast, every repeat sends its blame back to the same number, so it adds up
static void blame_add(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    TensorImpl& b = *t.parents[1];

    for (size_t i = 0; i < t.grad.size(); ++i) {
        size_t pa = source_position(i, t.shape, a.shape);  //which number of a made result[i]
        size_t pb = source_position(i, t.shape, b.shape);  //which number of b made result[i]

        if (a.requires_grad) a.grad[pa] += t.grad[i];
        if (b.requires_grad) b.grad[pb] += t.grad[i];
    }
}

//a * b: nudging a by 1 nudges the result by b, and nudging b by 1 nudges it by a
static void blame_mul(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    TensorImpl& b = *t.parents[1];

    std::vector<float> a_nums = values_of(a);
    std::vector<float> b_nums = values_of(b);

    for (size_t i = 0; i < t.grad.size(); ++i) {
        size_t pa = source_position(i, t.shape, a.shape);  //which number of a made result[i]
        size_t pb = source_position(i, t.shape, b.shape);  //which number of b made result[i]

        if (a.requires_grad) a.grad[pa] += t.grad[i] * b_nums[pb];
        if (b.requires_grad) b.grad[pb] += t.grad[i] * a_nums[pa];
    }
}

//sum: nudging any one number by 1 nudges the total by 1,
//so every number gets the blame unchanged
static void blame_sum(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];

    for (size_t i = 0; i < a.grad.size(); ++i) {
        a.grad[i] += t.grad[0];
    }
}

//mean: the total is divided by n, so nudging one number by 1 nudges the mean by 1/n
static void blame_mean(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    float n = static_cast<float>(a.grad.size());

    for (size_t i = 0; i < a.grad.size(); ++i) {
        a.grad[i] += t.grad[0] / n;
    }
}

//a - b: a gets the blame unchanged, b gets it flipped (nudging b up pushes the result down)
static void blame_sub(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    TensorImpl& b = *t.parents[1];

    for (size_t i = 0; i < t.grad.size(); ++i) {
        size_t pa = source_position(i, t.shape, a.shape);  //which number of a made result[i]
        size_t pb = source_position(i, t.shape, b.shape);  //which number of b made result[i]

        if (a.requires_grad) a.grad[pa] += t.grad[i];
        if (b.requires_grad) b.grad[pb] -= t.grad[i];
    }
}

//a / b: nudging a by 1 nudges the result by 1/b,
//nudging b by 1 nudges it by -a/b² (a bigger bottom number makes the result smaller)
static void blame_div(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    TensorImpl& b = *t.parents[1];

    std::vector<float> a_nums = values_of(a);
    std::vector<float> b_nums = values_of(b);

    for (size_t i = 0; i < t.grad.size(); ++i) {
        size_t pa = source_position(i, t.shape, a.shape);  //which number of a made result[i]
        size_t pb = source_position(i, t.shape, b.shape);  //which number of b made result[i]

        if (a.requires_grad) a.grad[pa] += t.grad[i] / b_nums[pb];
        if (b.requires_grad) b.grad[pb] -= t.grad[i] * a_nums[pa] / (b_nums[pb] * b_nums[pb]);
    }
}

//t + c (also t - c and +t): adding a fixed number doesn't change how a nudge spreads,
//so the blame passes through unchanged
static void blame_add_scalar(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];

    for (size_t i = 0; i < t.grad.size(); ++i) {
        a.grad[i] += t.grad[i];
    }
}

//t * c (also t / c and -t): nudging t by 1 nudges the result by c
static void blame_mul_scalar(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];

    for (size_t i = 0; i < t.grad.size(); ++i) {
        a.grad[i] += t.grad[i] * t.scalar;
    }
}

//c - t: nudging t up pushes the result down, so the blame is flipped
static void blame_scalar_sub(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];

    for (size_t i = 0; i < t.grad.size(); ++i) {
        a.grad[i] -= t.grad[i];
    }
}

//c / t: nudging t by 1 nudges the result by -c/t²
static void blame_scalar_div(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    std::vector<float> a_nums = values_of(a);

    for (size_t i = 0; i < t.grad.size(); ++i) {
        a.grad[i] -= t.grad[i] * t.scalar / (a_nums[i] * a_nums[i]);
    }
}

//relu(x): positive numbers passed straight through, so they get the blame;
//negative numbers were flattened to 0, so nudging them changes nothing
static void blame_relu(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    std::vector<float> a_nums = values_of(a);

    for (size_t i = 0; i < t.grad.size(); ++i) {
        if (a_nums[i] > 0.0f)
            a.grad[i] += t.grad[i];
    }
}

//exp(x): the slope of eˣ is eˣ itself, which is this tensor's own value
static void blame_exp(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    const float* out = t.storage->data() + t.offset;  //the results, eˣ

    for (size_t i = 0; i < t.grad.size(); ++i) {
        a.grad[i] += t.grad[i] * out[i];
    }
}

//log(x): nudging x by 1 nudges the result by 1/x (steep near 0, flat for big numbers)
static void blame_log(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    std::vector<float> a_nums = values_of(a);

    for (size_t i = 0; i < t.grad.size(); ++i) {
        a.grad[i] += t.grad[i] / a_nums[i];
    }
}

//tanh(x): the slope is 1 - tanh(x)², steep near 0 and flat where it squashes toward -1 or 1
static void blame_tanh(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    const float* out = t.storage->data() + t.offset;  //the results, tanh(x)

    for (size_t i = 0; i < t.grad.size(); ++i) {
        a.grad[i] += t.grad[i] * (1.0f - out[i] * out[i]);
    }
}

//matmul: C[i][j] adds up A[i][k] * B[k][j], so nudging A[i][k] by 1 moves every
//C[i][j] in row i by B[k][j]; its blame is the arriving blame times B, added up over j
//(and the same the other way round for B). In maths books: G·Bᵀ and Aᵀ·G
static void blame_matmul(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    TensorImpl& b = *t.parents[1];

    size_t n = a.shape[0];  //rows of a
    size_t m = a.shape[1];  //columns of a, rows of b
    size_t p = b.shape[1];  //columns of b

    std::vector<float> a_nums = values_of(a);
    std::vector<float> b_nums = values_of(b);

    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < p; ++col) {
            float g = t.grad[row * p + col];  //blame arriving at C[row][col]

            for (size_t k = 0; k < m; ++k) {
                if (a.requires_grad) a.grad[row * m + k] += g * b_nums[k * p + col];
                if (b.requires_grad) b.grad[k * p + col] += g * a_nums[row * m + k];
            }
        }
    }
}

//transpose: the numbers didn't change, only rows and columns swapped,
//so the blame is swapped back: a[i][j] gets the blame of result[j][i]
static void blame_transpose(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    size_t rows = a.shape[0];
    size_t cols = a.shape[1];

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            a.grad[i * cols + j] += t.grad[j * rows + i];
        }
    }
}

//reshape and contiguous: same numbers in the same reading order, just arranged
//differently, so the blame copies straight across
static void blame_same_order(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];

    for (size_t i = 0; i < t.grad.size(); ++i) {
        a.grad[i] += t.grad[i];
    }
}

//sum(axis): every number was added into one bucket (its column for axis 0, its row
//for axis 1), so each number gets its bucket's blame, spread back out
static void blame_sum_axis(TensorImpl& t) {
    TensorImpl& a = *t.parents[0];
    size_t rows = a.shape[0];
    size_t cols = a.shape[1];

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            size_t target = (t.axis == 0) ? col : row;  //the same bucket the forward pass used
            a.grad[row * cols + col] += t.grad[target];
        }
    }
}

//pass t's blame to the tensors it was made from, using t's operation
//(a tensor made by you, with op None, has nobody to pass blame to)
static void pass_blame_back(TensorImpl& t) {
    if (t.op == Op::Add)       blame_add(t);
    if (t.op == Op::Mul)       blame_mul(t);
    if (t.op == Op::Sum)       blame_sum(t);
    if (t.op == Op::Mean)      blame_mean(t);
    if (t.op == Op::Sub)       blame_sub(t);
    if (t.op == Op::Div)       blame_div(t);
    if (t.op == Op::AddScalar) blame_add_scalar(t);
    if (t.op == Op::MulScalar) blame_mul_scalar(t);
    if (t.op == Op::ScalarSub) blame_scalar_sub(t);
    if (t.op == Op::ScalarDiv) blame_scalar_div(t);
    if (t.op == Op::Relu)      blame_relu(t);
    if (t.op == Op::Exp)       blame_exp(t);
    if (t.op == Op::Log)       blame_log(t);
    if (t.op == Op::Tanh)      blame_tanh(t);
    if (t.op == Op::Matmul)    blame_matmul(t);
    if (t.op == Op::Transpose) blame_transpose(t);
    if (t.op == Op::Reshape || t.op == Op::Contiguous) blame_same_order(t);
    if (t.op == Op::SumAxis)   blame_sum_axis(t);
}

// fill in grad for every tracked tensor this one was made from
//e.g. chain = p * q + p, with p=3 and q=4: chain.backward() gives p.grad = [5], q.grad = [3]
void Tensor::backward() {

    //backward starts from the loss, which must be a single tracked number
    if (!impl_->requires_grad)
        throw std::invalid_argument("backward: tensor isn't tracked");
    if (numel() != 1)
        throw std::invalid_argument("backward: tensor must hold a single number");

    //1. list every tensor on the receipt, inputs before results
    std::vector<std::shared_ptr<TensorImpl>> order;
    std::set<TensorImpl*> visited;
    build_order(impl_, order, visited);

    //2. the loss is fully to blame for itself
    impl_->grad[0] = 1.0f;

    //3. walk the list backwards, from the loss to the inputs
    for (size_t i = order.size(); i-- > 0;) {
        pass_blame_back(*order[i]);
    }
}

// reset the blame to 0, ready for the next training step
//backward() adds blame with +=, so without this each step's blame piles on top of the last
void Tensor::zero_grad() {
    if (impl_->requires_grad)
        impl_->grad.assign(numel(), 0.0f);
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

    //if this tensor is tracked, write the receipt line for autograd (remembering the axis)
    if (impl_->requires_grad) {
        result.record(Op::SumAxis, {*this});
        result.impl_->axis = axis;
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

    //if either input is tracked, write the receipt line for autograd
    if (a.requires_grad() || b.requires_grad())
        result.record(Op::Add, {a, b});

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

    //if either input is tracked, write the receipt line for autograd
    if (a.requires_grad() || b.requires_grad())
        result.record(Op::Sub, {a, b});

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

    //if either input is tracked, write the receipt line for autograd
    if (a.requires_grad() || b.requires_grad())
        result.record(Op::Mul, {a, b});

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

    //if either input is tracked, write the receipt line for autograd
    if (a.requires_grad() || b.requires_grad())
        result.record(Op::Div, {a, b});

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

    //if the input is tracked, write the receipt line for autograd (-a is the same as a * -1)
    if (a.requires_grad())
        result.record(Op::MulScalar, {a}, -1.0f);

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

    //if the input is tracked, write the receipt line for autograd (+a is the same as a + 0)
    if (a.requires_grad())
        result.record(Op::AddScalar, {a}, 0.0f);

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

    //if the input is tracked, write the receipt line for autograd
    if (a.requires_grad())
        result.record(Op::AddScalar, {a}, b);

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

    //if the input is tracked, write the receipt line for autograd (t - c is the same as t + -c)
    if (a.requires_grad())
        result.record(Op::AddScalar, {a}, -b);

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

    //if the input is tracked, write the receipt line for autograd
    if (a.requires_grad())
        result.record(Op::MulScalar, {a}, b);

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

    //if the input is tracked, write the receipt line for autograd (t / c is the same as t * 1/c)
    if (a.requires_grad())
        result.record(Op::MulScalar, {a}, 1.0f / b);

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

    //if the tensor is tracked, write the receipt line for autograd
    if (b.requires_grad())
        result.record(Op::ScalarSub, {b});

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

    //if the tensor is tracked, write the receipt line for autograd (remembering c)
    if (b.requires_grad())
        result.record(Op::ScalarDiv, {b}, a);

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

    //if either input is tracked, write the receipt line for autograd
    if (a.requires_grad() || b.requires_grad())
        result.record(Op::Matmul, {a, b});

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

    //if the input is tracked, write the receipt line for autograd
    if (a.requires_grad())
        result.record(Op::Relu, {a});

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

    //if the input is tracked, write the receipt line for autograd
    if (a.requires_grad())
        result.record(Op::Exp, {a});

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

    //if the input is tracked, write the receipt line for autograd
    if (a.requires_grad())
        result.record(Op::Log, {a});

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

    //if the input is tracked, write the receipt line for autograd
    if (a.requires_grad())
        result.record(Op::Tanh, {a});

    return result;
}

// add up every number, as a single-number tensor autograd can track
//sum([1,2,3]) = [6]
Tensor sum(const Tensor& a) {
    Tensor result = Tensor::from_vector({a.sum()}, {1});

    //if the input is tracked, write the receipt line for autograd
    if (a.requires_grad())
        result.record(Op::Sum, {a});

    return result;
}

// the average of every number, as a single-number tensor autograd can track
//mean([1,2,3]) = [2]
Tensor mean(const Tensor& a) {
    Tensor result = Tensor::from_vector({a.mean()}, {1});

    //if the input is tracked, write the receipt line for autograd
    if (a.requires_grad())
        result.record(Op::Mean, {a});

    return result;
}

}  // namespace nnf


