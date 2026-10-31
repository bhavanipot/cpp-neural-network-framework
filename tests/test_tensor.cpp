// Tests for nnf::Tensor. Each check stays quiet when the answer is right,
// and prints FAIL when it's wrong. Run all of them with:
//   cmake --build build && ctest --test-dir build --output-on-failure

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "nnf/tensor.h"

using nnf::Tensor;

static int failures = 0;  //count of how many checks failed

//record a failed check
static void fail(const std::string& name) {
    ++failures;
    std::cout << "FAIL: " << name << "\n";
}

//check two numbers are equal
static void check_number(const std::string& name, float got, float expected) {
    if (std::fabs(got - expected) > 1e-4f)
        fail(name + ", got " + std::to_string(got));
}

//check a tensor has the expected shape and numbers
static void check_tensor(const std::string& name, const Tensor& t,
                         const nnf::Shape& shape, const std::vector<float>& expected) {
    if (t.shape() != shape) {
        fail(name + ", wrong shape: " + t.to_string());
        return;
    }

    Tensor ordered = t.contiguous();  //so transposed tensors can be checked too
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(ordered.data()[i] - expected[i]) > 1e-4f) {
            fail(name + ", got " + t.to_string());
            return;
        }
    }
}

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

    if (failures == 0)
        std::cout << "all tests passed\n";
    else
        std::cout << failures << " check(s) failed\n";

    return failures == 0 ? 0 : 1;  //non-zero tells ctest something failed
}
