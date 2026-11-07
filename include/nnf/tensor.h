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

//which operation made a tensor, so backward() knows which rule to use
enum class Op { None, Add, Mul, Sum, Mean, Sub, Div, AddScalar, MulScalar, ScalarSub, ScalarDiv,
                Relu, Exp, Log, Tanh, Matmul, Transpose, Reshape, Contiguous, SumAxis };

std::string op_name(Op op); //e.g. "Add" or "Mul", for printing

struct TensorImpl {
    std::shared_ptr<std::vector<float>> storage;  // all numbers, in one flat list
    Shape shape;                                  // e.g. {2, 3}
    std::vector<size_t> strides;                  // e.g. {3, 1}
    size_t offset = 0;                            // where this tensor starts in storage

    bool requires_grad = false;                   // should autograd track this tensor?
    std::vector<float> grad;                      // the blame for each number, filled in by backward()

    Op op = Op::None;                                  // which operation made this tensor
    std::vector<std::shared_ptr<TensorImpl>> parents;  // the tensors it was made from
    float scalar = 0.0f;                               // the plain number in a scalar op, e.g. the 2 in t * 2
    size_t axis = 0;                                   // the axis in sum(axis), so backward knows which way was collapsed
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

    float sum() const;  //add up every number into one total
    float mean() const; //the average of every number
    Tensor sum(size_t axis) const;  //add up along one axis of a 2-D tensor, e.g. {2,3} on axis 1 -> {2}
    Tensor mean(size_t axis) const; //average along one axis of a 2-D tensor

    float* data() { return impl_->storage->data() + impl_->offset; }  //pointer to the numbers
    const float* data() const { return impl_->storage->data() + impl_->offset; } //read-only version

    float& at(const std::vector<size_t>& index); //finding value at index

    Tensor reshape(const Shape& new_shape) const; //same data, different shape
    Tensor transpose() const; //swap rows and columns of a 2-D tensor, no copy
    Tensor expand(const Shape& new_shape) const; //repeat to a bigger shape, no copy
    bool is_contiguous() const; //true if the numbers are stored in reading order
    Tensor contiguous() const; //an in-order copy, or itself if already in order

    void set_requires_grad(bool on); //start or stop tracking this tensor for autograd
    bool requires_grad() const { return impl_->requires_grad; } //is it being tracked?
    Tensor grad() const; //the blame for each number, as a tensor of the same shape

    void record(Op op, const std::vector<Tensor>& inputs, float scalar = 0.0f); //write the receipt line, used by operations
    Op op() const { return impl_->op; } //which operation made this tensor
    size_t num_parents() const { return impl_->parents.size(); } //how many inputs made it
    Tensor parent(size_t i) const; //one of the inputs that made it

    void backward(); //fill in grad for every tracked tensor this one was made from
    void zero_grad(); //reset the blame to 0, ready for the next training step

private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<TensorImpl> impl_;
};


Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a); // unary minus
Tensor operator+(const Tensor& a); // unary plus

// tensor-scalar math, e.g. t * 2.0f
Tensor operator+(const Tensor& a, float b);
Tensor operator-(const Tensor& a, float b);
Tensor operator*(const Tensor& a, float b);
Tensor operator/(const Tensor& a, float b);

// scalar-tensor math, e.g. 1.0f - t
Tensor operator+(float a, const Tensor& b);
Tensor operator-(float a, const Tensor& b);
Tensor operator*(float a, const Tensor& b);
Tensor operator/(float a, const Tensor& b);

// matrix multiply: {n, m} x {m, p} -> {n, p}
Tensor matmul(const Tensor& a, const Tensor& b);

// elementwise functions, applied to every number
Tensor relu(const Tensor& a); //negatives become 0
Tensor exp(const Tensor& a);  //e to the power of each number
Tensor log(const Tensor& a);  //natural log of each number
Tensor tanh(const Tensor& a); //every number has a domain of (-1, 1)

// sum and mean that autograd can track, giving a single-number tensor
//(t.sum() and t.mean() give a plain float instead)
Tensor sum(const Tensor& a);  //sum([1,2,3]) = [6]
Tensor mean(const Tensor& a); //mean([1,2,3]) = [2]

}  // namespace nnf
