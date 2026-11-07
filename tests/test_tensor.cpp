// Tests for nnf::Tensor. Each check stays quiet when the answer is right,
// and prints FAIL when it's wrong. Run all of them with:
//   cmake --build build && ctest --test-dir build --output-on-failure

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "test_helpers.h"  //fail, check_number, check_tensor, report

static void test_creation() {
    check_tensor("zeros", Tensor::zeros({2, 3}), {2, 3}, {0, 0, 0, 0, 0, 0});
    check_tensor("full", Tensor::full({2, 2}, 7.0f), {2, 2}, {7, 7, 7, 7});
    check_tensor("full with no dimensions", Tensor::full({}, 5.0f), {}, {5});
    check_tensor("from_vector", Tensor::from_vector({1, 2, 3, 4}, {2, 2}), {2, 2}, {1, 2, 3, 4});

    //the same seed must give the same numbers, so the difference is all zeros
    Tensor r1 = Tensor::randn({2, 3}, 7);
    Tensor r2 = Tensor::randn({2, 3}, 7);
    check_tensor("randn is repeatable", r1 - r2, {2, 3}, {0, 0, 0, 0, 0, 0});

    try {
        Tensor::from_vector({1, 2, 3}, {2, 2});
        fail("from_vector with the wrong size should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_at() {
    Tensor t = Tensor::zeros({2, 3});
    t.at({1, 2}) = 5.0f;
    check_number("at write then read", t.at({1, 2}), 5.0f);
    check_number("at other cell untouched", t.at({0, 0}), 0.0f);

    try {
        t.at({5, 0});
        fail("at out of range should throw");
    } catch (const std::out_of_range&) {}

    try {
        t.at({1});
        fail("at with too few indices should throw");
    } catch (const std::out_of_range&) {}
}

static void test_math() {
    Tensor a = Tensor::from_vector({1, 2, 3}, {3});
    Tensor b = Tensor::from_vector({4, 5, 6}, {3});

    check_tensor("add", a + b, {3}, {5, 7, 9});
    check_tensor("subtract", a - b, {3}, {-3, -3, -3});
    check_tensor("multiply", a * b, {3}, {4, 10, 18});
    check_tensor("divide", a / b, {3}, {0.25f, 0.4f, 0.5f});
    check_tensor("unary minus", -a, {3}, {-1, -2, -3});
    check_tensor("unary plus", +a, {3}, {1, 2, 3});
}

static void test_scalar_math() {
    Tensor a = Tensor::from_vector({1, 2, 3}, {3});

    check_tensor("tensor + number", a + 10.0f, {3}, {11, 12, 13});
    check_tensor("tensor - number", a - 10.0f, {3}, {-9, -8, -7});
    check_tensor("tensor * number", a * 10.0f, {3}, {10, 20, 30});
    check_tensor("tensor / number", a / 10.0f, {3}, {0.1f, 0.2f, 0.3f});
    check_tensor("number + tensor", 10.0f + a, {3}, {11, 12, 13});
    check_tensor("number - tensor", 10.0f - a, {3}, {9, 8, 7});
    check_tensor("number * tensor", 10.0f * a, {3}, {10, 20, 30});
    check_tensor("number / tensor", 10.0f / a, {3}, {10, 5, 3.33333f});
}

static void test_sum_and_mean() {
    Tensor grid = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});

    check_number("sum", grid.sum(), 21.0f);
    check_number("mean", grid.mean(), 3.5f);
    check_tensor("sum axis 0", grid.sum(0), {3}, {5, 7, 9});
    check_tensor("sum axis 1", grid.sum(1), {2}, {6, 15});
    check_tensor("mean axis 0", grid.mean(0), {3}, {2.5f, 3.5f, 4.5f});
    check_tensor("mean axis 1", grid.mean(1), {2}, {2, 5});

    try {
        Tensor::zeros({0}).mean();
        fail("mean of an empty tensor should throw");
    } catch (const std::invalid_argument&) {}

    try {
        grid.sum(2);
        fail("sum on an axis that doesn't exist should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_matmul() {
    Tensor x = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor y = Tensor::from_vector({7, 8, 9, 10, 11, 12}, {3, 2});

    check_tensor("matmul", nnf::matmul(x, y), {2, 2}, {58, 64, 139, 154});

    try {
        nnf::matmul(x, x);
        fail("matmul with mismatched inner sizes should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_reshape() {
    Tensor flat = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {6});
    Tensor grid = flat.reshape({2, 3});
    check_tensor("reshape", grid, {2, 3}, {1, 2, 3, 4, 5, 6});

    //reshape shares storage, so a change to one shows up in the other
    grid.at({0, 0}) = 99.0f;
    check_number("reshape shares storage", flat.at({0}), 99.0f);

    try {
        flat.reshape({2, 2});
        fail("reshape to a different size should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_transpose_and_contiguous() {
    Tensor mat = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor mat_t = mat.transpose();

    check_tensor("transpose", mat_t, {3, 2}, {1, 4, 2, 5, 3, 6});
    check_number("transpose at", mat_t.at({0, 1}), 4.0f);
    check_number("original is contiguous", mat.is_contiguous(), 1.0f);
    check_number("transpose is not contiguous", mat_t.is_contiguous(), 0.0f);

    //contiguous makes an in-order copy that math can use
    Tensor mat_tc = mat_t.contiguous();
    check_number("contiguous copy is contiguous", mat_tc.is_contiguous(), 1.0f);
    check_tensor("matmul with contiguous transpose", nnf::matmul(mat, mat_tc), {2, 2}, {14, 32, 32, 77});

    //transpose shares storage, so a change to one shows up in the other
    mat_t.at({2, 0}) = 30.0f;
    check_number("transpose shares storage", mat.at({0, 2}), 30.0f);

    try {
        nnf::matmul(mat, mat_t);
        fail("matmul with a transposed tensor should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_functions() {
    Tensor v = Tensor::from_vector({-2, 0, 1, 2}, {4});

    check_tensor("relu", nnf::relu(v), {4}, {0, 0, 1, 2});
    check_tensor("exp", nnf::exp(v), {4}, {0.135335f, 1, 2.71828f, 7.38906f});
    check_tensor("tanh", nnf::tanh(v), {4}, {-0.964028f, 0, 0.761594f, 0.964028f});
    check_tensor("log undoes exp", nnf::log(nnf::exp(v)), {4}, {-2, 0, 1, 2});
}

static void test_broadcasting() {
    Tensor grid = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor bias = Tensor::from_vector({10, 20, 30}, {3});
    Tensor column = Tensor::from_vector({100, 200}, {2, 1});

    check_tensor("row added to every row", grid + bias, {2, 3}, {11, 22, 33, 14, 25, 36});
    check_tensor("order doesn't matter", bias + grid, {2, 3}, {11, 22, 33, 14, 25, 36});
    check_tensor("column added to every column", grid + column, {2, 3}, {101, 102, 103, 204, 205, 206});

    //divide each row by its own total, so every row adds up to 1
    Tensor probs = grid / grid.sum(1).reshape({2, 1});
    check_tensor("rows divided by totals", probs, {2, 3}, {1 / 6.0f, 2 / 6.0f, 3 / 6.0f, 4 / 15.0f, 5 / 15.0f, 6 / 15.0f});
    check_tensor("each row adds up to 1", probs.sum(1), {2}, {1, 1});

    //+ - * / put a transposed tensor in order first
    Tensor mat_t = grid.transpose();
    check_tensor("add transposed tensors", mat_t + mat_t, {3, 2}, {2, 8, 4, 10, 6, 12});

    try {
        grid + Tensor::zeros({2});
        fail("adding shapes {2,3} and {2} should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_tracking() {
    Tensor w = Tensor::from_vector({1, 2, 3}, {3});
    check_number("starts untracked", w.requires_grad(), 0.0f);

    w.set_requires_grad(true);
    check_number("tracked after turning on", w.requires_grad(), 1.0f);
    check_tensor("grad starts at zero", w.grad(), {3}, {0, 0, 0});

    Tensor w2 = w;  //same tensor underneath
    check_number("copies share tracking", w2.requires_grad(), 1.0f);

    w.set_requires_grad(false);
    try {
        w.grad();
        fail("grad of an untracked tensor should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_receipt() {
    Tensor p = Tensor::from_vector({3}, {1});
    Tensor q = Tensor::from_vector({4}, {1});
    p.set_requires_grad(true);
    q.set_requires_grad(true);

    //p * q writes a Mul line pointing at p and q
    Tensor y = p * q;
    check_number("mul is recorded", y.op() == nnf::Op::Mul, 1.0f);
    check_number("mul has 2 parents", y.num_parents(), 2.0f);
    check_number("mul parent 0 is p", y.parent(0).data() == p.data(), 1.0f);  //same numbers, not a copy
    check_number("mul parent 1 is q", y.parent(1).data() == q.data(), 1.0f);
    check_number("mul result is tracked", y.requires_grad(), 1.0f);

    //the receipt grows as results are used again
    Tensor z = y + p;
    check_number("add is recorded", z.op() == nnf::Op::Add, 1.0f);
    check_number("add parent 0 is y", z.parent(0).data() == y.data(), 1.0f);

    //untracked inputs write nothing
    Tensor a = Tensor::from_vector({1, 2}, {2});
    Tensor b = Tensor::from_vector({3, 4}, {2});
    Tensor c = a + b;
    check_number("untracked is not recorded", c.op() == nnf::Op::None, 1.0f);
    check_number("untracked has no parents", c.num_parents(), 0.0f);
    check_number("untracked result stays untracked", c.requires_grad(), 0.0f);

    //one tracked input is enough to record
    Tensor mixed = p + Tensor::from_vector({1}, {1});
    check_number("tracked plus untracked is recorded", mixed.op() == nnf::Op::Add, 1.0f);

    //op_name turns the operation into text for printing
    if (nnf::op_name(nnf::Op::Mul) != "Mul") fail("op_name of Mul");
    if (nnf::op_name(nnf::Op::None) != "None") fail("op_name of None");

    try {
        y.parent(5);
        fail("parent out of range should throw");
    } catch (const std::out_of_range&) {}
}

//make a tracked tensor holding one number
static Tensor tracked(float value) {
    Tensor t = Tensor::from_vector({value}, {1});
    t.set_requires_grad(true);
    return t;
}

static void test_backward() {
    //p * q: p's blame is q, q's blame is p
    Tensor p = tracked(3);
    Tensor q = tracked(4);
    (p * q).backward();
    check_tensor("p * q, blame on p", p.grad(), {1}, {4});
    check_tensor("p * q, blame on q", q.grad(), {1}, {3});

    //p * q + p: p is used twice, so its blame adds up to q + 1
    Tensor p2 = tracked(3);
    Tensor q2 = tracked(4);
    (p2 * q2 + p2).backward();
    check_tensor("p * q + p, blame on p", p2.grad(), {1}, {5});
    check_tensor("p * q + p, blame on q", q2.grad(), {1}, {3});

    //p * p: both inputs are p, so it gets p + p
    Tensor p3 = tracked(3);
    (p3 * p3).backward();
    check_tensor("p * p, blame on p", p3.grad(), {1}, {6});

    //p * c with c untracked: only p gets blame, and nothing crashes
    Tensor p4 = tracked(3);
    Tensor c = Tensor::from_vector({7}, {1});
    (p4 * c).backward();
    check_tensor("p * untracked, blame on p", p4.grad(), {1}, {7});

    try {
        Tensor::from_vector({1}, {1}).backward();
        fail("backward on an untracked tensor should throw");
    } catch (const std::invalid_argument&) {}

    try {
        Tensor v = Tensor::from_vector({1, 2}, {2});
        v.set_requires_grad(true);
        v.backward();
        fail("backward on a tensor with 2 numbers should throw");
    } catch (const std::invalid_argument&) {}
}

//any function that takes a list of tensors and gives back a tensor
using TestFunction = Tensor (*)(const std::vector<Tensor>& x);

//calculations for the gradient checker, each builds one number from its inputs
static Tensor f_add(const std::vector<Tensor>& x)    { return x[0] + x[1]; }
static Tensor f_mul(const std::vector<Tensor>& x)    { return x[0] * x[1]; }
static Tensor f_square(const std::vector<Tensor>& x) { return x[0] * x[0]; }
static Tensor f_chain(const std::vector<Tensor>& x)  { return x[0] * x[1] + x[0]; }
static Tensor f_long(const std::vector<Tensor>& x)   { return (x[0] + x[1]) * (x[0] * x[2]) + x[2]; }
static Tensor f_sum(const std::vector<Tensor>& x)         { return nnf::sum(x[0]); }
static Tensor f_mean(const std::vector<Tensor>& x)        { return nnf::mean(x[0]); }
static Tensor f_neuron(const std::vector<Tensor>& x)      { return nnf::sum(x[0] * x[1]); }
static Tensor f_mean_square(const std::vector<Tensor>& x) { return nnf::mean(x[0] * x[0]); }
static Tensor f_sum_mixed(const std::vector<Tensor>& x)   { return nnf::sum(x[0] * x[1] + x[0]); }
static Tensor f_sub(const std::vector<Tensor>& x)           { return nnf::sum(x[0] - x[1]); }
static Tensor f_div(const std::vector<Tensor>& x)           { return nnf::sum(x[0] / x[1]); }
static Tensor f_neg(const std::vector<Tensor>& x)           { return nnf::sum(-x[0]); }
static Tensor f_pos(const std::vector<Tensor>& x)           { return nnf::sum(+x[0]); }
static Tensor f_plus_number(const std::vector<Tensor>& x)   { return nnf::sum(x[0] + 3.0f); }
static Tensor f_minus_number(const std::vector<Tensor>& x)  { return nnf::sum(x[0] - 3.0f); }
static Tensor f_times_number(const std::vector<Tensor>& x)  { return nnf::sum(x[0] * 2.0f); }
static Tensor f_over_number(const std::vector<Tensor>& x)   { return nnf::sum(x[0] / 4.0f); }
static Tensor f_number_plus(const std::vector<Tensor>& x)   { return nnf::sum(3.0f + x[0]); }
static Tensor f_number_times(const std::vector<Tensor>& x)  { return nnf::sum(2.0f * x[0]); }
static Tensor f_number_minus(const std::vector<Tensor>& x)  { return nnf::sum(10.0f - x[0]); }
static Tensor f_number_over(const std::vector<Tensor>& x)   { return nnf::sum(6.0f / x[0]); }
static Tensor f_squared_error(const std::vector<Tensor>& x) { return nnf::mean((x[0] - x[1]) * (x[0] - x[1])); }
static Tensor f_relu(const std::vector<Tensor>& x)          { return nnf::sum(nnf::relu(x[0])); }
static Tensor f_exp(const std::vector<Tensor>& x)           { return nnf::sum(nnf::exp(x[0])); }
static Tensor f_log(const std::vector<Tensor>& x)           { return nnf::sum(nnf::log(x[0])); }
static Tensor f_tanh(const std::vector<Tensor>& x)          { return nnf::sum(nnf::tanh(x[0])); }
static Tensor f_activations(const std::vector<Tensor>& x)   { return nnf::mean(nnf::tanh(x[0] * x[1]) + nnf::relu(x[0])); }
static Tensor f_matmul(const std::vector<Tensor>& x)            { return nnf::sum(nnf::matmul(x[0], x[1])); }
static Tensor f_matmul_tanh(const std::vector<Tensor>& x)       { return nnf::mean(nnf::tanh(nnf::matmul(x[0], x[1]))); }
static Tensor f_matmul_transposed(const std::vector<Tensor>& x) { return nnf::sum(nnf::matmul(x[0], x[1].transpose().contiguous())); }
static Tensor f_transpose_mul(const std::vector<Tensor>& x)     { return nnf::sum(x[0].transpose() * x[1]); }
static Tensor f_reshape_mul(const std::vector<Tensor>& x)       { return nnf::sum(x[0].reshape({3, 2}) * x[1]); }
static Tensor f_add_broadcast(const std::vector<Tensor>& x)  { return nnf::sum((x[0] + x[1]) * x[2]); }
static Tensor f_sub_row(const std::vector<Tensor>& x)        { return nnf::sum((x[0] - x[1]) * x[2]); }
static Tensor f_row_sub(const std::vector<Tensor>& x)        { return nnf::sum((x[1] - x[0]) * x[2]); }
static Tensor f_mul_row(const std::vector<Tensor>& x)        { return nnf::sum(x[0] * x[1] * x[2]); }
static Tensor f_div_column(const std::vector<Tensor>& x)     { return nnf::sum(x[0] / x[1] * x[2]); }
static Tensor f_sum_axis0(const std::vector<Tensor>& x)      { return nnf::sum(x[0].sum(0) * x[1]); }
static Tensor f_sum_axis1(const std::vector<Tensor>& x)      { return nnf::sum(x[0].sum(1) * x[1]); }
static Tensor f_mean_axis1(const std::vector<Tensor>& x)     { return nnf::sum(x[0].mean(1) * x[1]); }
static Tensor f_softmax(const std::vector<Tensor>& x)        { return nnf::sum(nnf::exp(x[0]) / nnf::exp(x[0]).sum(1).reshape({2, 1}) * x[1]); }
static Tensor f_layer(const std::vector<Tensor>& x)          { return nnf::mean(nnf::tanh(nnf::matmul(x[0], x[1]) + x[2])); }

//a fresh, untracked copy of t's numbers
static Tensor copy_of(const Tensor& t) {
    Tensor ordered = t.contiguous();
    std::vector<float> nums(ordered.data(), ordered.data() + ordered.numel());
    return Tensor::from_vector(nums, t.shape());
}

//compare backward()'s blame with the slow way: nudge each number up and down and measure
static void check_gradient(const std::string& name, TestFunction f, const std::vector<Tensor>& inputs) {

    //the fast way: track copies of the inputs, run once, then backward
    std::vector<Tensor> tracked_inputs;
    for (const Tensor& x : inputs) {
        Tensor copy = copy_of(x);
        copy.set_requires_grad(true);
        tracked_inputs.push_back(copy);
    }
    f(tracked_inputs).backward();

    //the slow way: for every number in every input...
    const float h = 0.01f;  //the size of the nudge

    for (size_t k = 0; k < inputs.size(); ++k) {
        Tensor fast_blame = tracked_inputs[k].grad();

        for (size_t i = 0; i < inputs[k].numel(); ++i) {

            //...run once with it nudged up, once nudged down
            std::vector<Tensor> up;
            std::vector<Tensor> down;
            for (const Tensor& x : inputs) {
                up.push_back(copy_of(x));
                down.push_back(copy_of(x));
            }
            up[k].data()[i] += h;
            down[k].data()[i] -= h;

            float slow = (f(up).sum() - f(down).sum()) / (2 * h);
            float fast = fast_blame.data()[i];

            //they should agree to within about 1%
            if (std::fabs(fast - slow) > 0.01f * std::max(1.0f, std::fabs(slow)))
                fail(name + ": input " + std::to_string(k) + ", number " + std::to_string(i) +
                     ": backward gave " + std::to_string(fast) + ", nudging gave " + std::to_string(slow));
        }
    }
}

static void test_gradient_checker() {
    Tensor two = Tensor::from_vector({2}, {1});
    Tensor minus_three = Tensor::from_vector({-3}, {1});
    Tensor half = Tensor::from_vector({0.5f}, {1});

    check_gradient("add", f_add, {two, minus_three});
    check_gradient("mul", f_mul, {two, minus_three});
    check_gradient("square", f_square, {minus_three});
    check_gradient("chain", f_chain, {two, minus_three});
    check_gradient("long chain", f_long, {two, minus_three, half});
}

static void test_sum_mean_backward() {
    //the values match the float versions
    Tensor a = Tensor::from_vector({1, 2, 3}, {3});
    check_tensor("tracked sum value", nnf::sum(a), {1}, {6});
    check_tensor("tracked mean value", nnf::mean(a), {1}, {2});
    check_number("untracked sum is not recorded", nnf::sum(a).op() == nnf::Op::None, 1.0f);

    //sum: every number gets the blame unchanged
    Tensor w = Tensor::from_vector({1, 2, 3}, {3});
    w.set_requires_grad(true);
    check_number("sum is recorded", nnf::sum(w).op() == nnf::Op::Sum, 1.0f);
    nnf::sum(w).backward();
    check_tensor("sum blame", w.grad(), {3}, {1, 1, 1});

    //mean: every number gets the blame divided by n
    Tensor m = Tensor::from_vector({1, 2, 3}, {3});
    m.set_requires_grad(true);
    nnf::mean(m).backward();
    check_tensor("mean blame", m.grad(), {3}, {1 / 3.0f, 1 / 3.0f, 1 / 3.0f});

    //the slow way agrees on whole vectors
    Tensor x = Tensor::from_vector({1, -2, 3}, {3});
    Tensor y = Tensor::from_vector({0.5f, 4, -1}, {3});
    check_gradient("sum", f_sum, {x});
    check_gradient("mean", f_mean, {x});
    check_gradient("neuron", f_neuron, {x, y});
    check_gradient("mean square", f_mean_square, {x});
    check_gradient("sum of mixed", f_sum_mixed, {x, y});
}

static void test_arithmetic_backward() {
    //the squared error example from the demo: up, down, leave alone
    Tensor pred = Tensor::from_vector({2.5f, 0, 2}, {3});
    Tensor target = Tensor::from_vector({3, -0.5f, 2}, {3});
    pred.set_requires_grad(true);
    Tensor diff = pred - target;
    nnf::mean(diff * diff).backward();
    check_tensor("squared error blame", pred.grad(), {3}, {-1 / 3.0f, 1 / 3.0f, 0});

    //the slow way agrees for every operation (y has no zeros, so dividing by it is safe)
    Tensor x = Tensor::from_vector({1, -2, 3}, {3});
    Tensor y = Tensor::from_vector({0.5f, 4, -1}, {3});
    check_gradient("a - b", f_sub, {x, y});
    check_gradient("a / b", f_div, {x, y});
    check_gradient("-a", f_neg, {x});
    check_gradient("+a", f_pos, {x});
    check_gradient("t + 3", f_plus_number, {x});
    check_gradient("t - 3", f_minus_number, {x});
    check_gradient("t * 2", f_times_number, {x});
    check_gradient("t / 4", f_over_number, {x});
    check_gradient("3 + t", f_number_plus, {x});
    check_gradient("2 * t", f_number_times, {x});
    check_gradient("10 - t", f_number_minus, {x});
    check_gradient("6 / t", f_number_over, {y});
    check_gradient("squared error", f_squared_error, {x, y});
}

static void test_activation_backward() {
    //relu blocks the blame where the input was negative
    Tensor h = Tensor::from_vector({-1, 0.5f, 2}, {3});
    h.set_requires_grad(true);
    nnf::sum(nnf::relu(h)).backward();
    check_tensor("relu blame", h.grad(), {3}, {0, 1, 1});

    //the slow way agrees (no zeros, where relu has its corner; positive numbers for log)
    Tensor x = Tensor::from_vector({1, -2, 3}, {3});
    Tensor y = Tensor::from_vector({0.5f, 4, -1}, {3});
    Tensor positive = Tensor::from_vector({0.5f, 4, 2}, {3});
    check_gradient("relu", f_relu, {x});
    check_gradient("exp", f_exp, {x});
    check_gradient("log", f_log, {positive});
    check_gradient("tanh", f_tanh, {x});
    check_gradient("activations chained", f_activations, {x, y});
}

static void test_matmul_backward() {
    //a {1,2} times b {2,1}: a's blame is b's numbers, b's blame is a's numbers
    Tensor a = Tensor::from_vector({1, 2}, {1, 2});
    Tensor b = Tensor::from_vector({3, 4}, {2, 1});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    nnf::sum(nnf::matmul(a, b)).backward();  // 1×3 + 2×4 = 11
    check_tensor("matmul blame on a", a.grad(), {1, 2}, {3, 4});
    check_tensor("matmul blame on b", b.grad(), {2, 1}, {1, 2});

    //contiguous only writes a receipt when it actually copies
    Tensor w = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    w.set_requires_grad(true);
    check_number("transpose is recorded", w.transpose().op() == nnf::Op::Transpose, 1.0f);
    check_number("copying contiguous is recorded", w.transpose().contiguous().op() == nnf::Op::Contiguous, 1.0f);
    check_number("in-order contiguous is the same tensor", w.contiguous().data() == w.data(), 1.0f);

    //the slow way agrees; multiplying by a second tensor gives every position a
    //different blame, so a mix-up in ordering would show up
    Tensor m23 = Tensor::from_vector({1, -2, 3, 0.5f, 4, -1}, {2, 3});
    Tensor m32 = Tensor::from_vector({2, -1, 0.5f, 3, -2, 1}, {3, 2});
    Tensor m23b = Tensor::from_vector({0.3f, 1, -1, 2, -0.5f, 1.5f}, {2, 3});
    check_gradient("matmul", f_matmul, {m23, m32});
    check_gradient("matmul then tanh", f_matmul_tanh, {m23, m32});
    check_gradient("matmul with a transposed weight", f_matmul_transposed, {m23, m23b});
    check_gradient("transpose times a tensor", f_transpose_mul, {m23, m32});
    check_gradient("reshape times a tensor", f_reshape_mul, {m23, m32});
}

static void test_broadcasting_backward() {
    //the bias is used by both examples, so its blame adds up from both rows
    Tensor xb = Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    Tensor wb = Tensor::from_vector({1, 0, 0, 1}, {2, 2});
    Tensor bb = Tensor::from_vector({0.5f, -0.5f}, {2});
    wb.set_requires_grad(true);
    bb.set_requires_grad(true);
    nnf::sum(nnf::matmul(xb, wb) + bb).backward();
    check_tensor("bias blame adds up over the rows", bb.grad(), {2}, {2, 2});
    check_tensor("weight blame with a bias", wb.grad(), {2, 2}, {4, 4, 6, 6});

    //the slow way agrees; multiplying by w gives every position a different blame
    Tensor m = Tensor::from_vector({1, -2, 3, 0.5f, 4, -1}, {2, 3});
    Tensor w = Tensor::from_vector({0.3f, 1, -1, 2, -0.5f, 1.5f}, {2, 3});
    Tensor row = Tensor::from_vector({0.5f, -1, 2}, {3});
    Tensor column = Tensor::from_vector({2, -1.5f}, {2, 1});
    Tensor w3 = Tensor::from_vector({1, -2, 0.5f}, {3});
    Tensor w2 = Tensor::from_vector({3, -1}, {2});
    Tensor weights = Tensor::from_vector({2, -1, 0.5f, 3, -2, 1}, {3, 2});
    Tensor bias = Tensor::from_vector({0.1f, -0.2f}, {2});

    check_gradient("add a row", f_add_broadcast, {m, row, w});
    check_gradient("add a column", f_add_broadcast, {m, column, w});
    check_gradient("subtract a row", f_sub_row, {m, row, w});
    check_gradient("row minus a matrix", f_row_sub, {m, row, w});
    check_gradient("multiply by a row", f_mul_row, {m, row, w});
    check_gradient("divide by a column", f_div_column, {m, column, w});
    check_gradient("sum axis 0", f_sum_axis0, {m, w3});
    check_gradient("sum axis 1", f_sum_axis1, {m, w2});
    check_gradient("mean axis 1", f_mean_axis1, {m, w2});
    check_gradient("softmax", f_softmax, {m, w});
    check_gradient("layer with a bias", f_layer, {m, weights, bias});
}

static void test_zero_grad() {
    //backward adds blame with +=, so two backwards pile up
    Tensor zg = Tensor::from_vector({3}, {1});
    zg.set_requires_grad(true);
    (zg * 2.0f).backward();
    (zg * 2.0f).backward();
    check_tensor("blame piles up without zero_grad", zg.grad(), {1}, {4});

    zg.zero_grad();
    check_tensor("zero_grad clears the blame", zg.grad(), {1}, {0});

    //an untracked tensor has no blame to clear, so nothing happens
    Tensor plain = Tensor::from_vector({1, 2}, {2});
    plain.zero_grad();
    check_number("zero_grad leaves untracked alone", plain.requires_grad(), 0.0f);
}

int main() {
    test_creation();
    test_at();
    test_math();
    test_scalar_math();
    test_sum_and_mean();
    test_matmul();
    test_reshape();
    test_transpose_and_contiguous();
    test_functions();
    test_broadcasting();
    test_tracking();
    test_receipt();
    test_backward();
    test_gradient_checker();
    test_sum_mean_backward();
    test_arithmetic_backward();
    test_activation_backward();
    test_matmul_backward();
    test_broadcasting_backward();
    test_zero_grad();

    return report();
}
