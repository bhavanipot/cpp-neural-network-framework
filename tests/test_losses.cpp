// Tests for softmax, cross-entropy, one_hot and argmax.
// Run with: cmake --build build && ctest --test-dir build --output-on-failure

#include <stdexcept>
#include <string>
#include <vector>
#include "test_helpers.h"  //fail, check_number, check_tensor, report
#include "nnf/layers.h"
#include "nnf/losses.h"
#include "nnf/optim.h"

static void test_softmax() {
    //[2, 1, 0.1]: exp is [7.39, 2.72, 1.11], total 11.21
    Tensor probs = nnf::softmax(Tensor::from_vector({2, 1, 0.1f}, {1, 3}));
    check_number("softmax class 0", probs.data()[0], 0.659f, 0.001f);
    check_number("softmax class 1", probs.data()[1], 0.242f, 0.001f);
    check_number("softmax class 2", probs.data()[2], 0.099f, 0.001f);
    check_number("softmax row adds up to 1", probs.sum(), 1.0f);

    //exp(1000) would be infinity, subtracting the max keeps it safe
    Tensor huge = nnf::softmax(Tensor::from_vector({1000, 1001, 1002}, {1, 3}));
    check_number("huge softmax class 0", huge.data()[0], 0.090f, 0.001f);
    check_number("huge softmax class 1", huge.data()[1], 0.245f, 0.001f);
    check_number("huge softmax class 2", huge.data()[2], 0.665f, 0.001f);
}

static void test_one_hot_and_argmax() {
    check_tensor("one_hot", nnf::one_hot({2, 0}, 3), {2, 3}, {0, 0, 1, 1, 0, 0});

    //answer 5 doesn't exist when there are only 3 classes
    try {
        nnf::one_hot({5}, 3);
        fail("one_hot with an answer that isn't a class should throw");
    } catch (const std::invalid_argument&) {}

    std::vector<size_t> guesses = nnf::argmax(Tensor::from_vector({2, 1, 0.1f, 0.5f, 2.5f, 0.3f}, {2, 3}));
    check_number("argmax row 0", guesses[0], 0.0f);
    check_number("argmax row 1", guesses[1], 1.0f);
}

static void test_cross_entropy_value() {
    //(−log(0.659) − log(0.803)) ÷ 2 = (0.417 + 0.220) ÷ 2
    Tensor logits = Tensor::from_vector({2, 1, 0.1f, 0.5f, 2.5f, 0.3f}, {2, 3});
    check_number("cross_entropy value", nnf::cross_entropy(logits, {0, 1}).sum(), 0.3185f, 0.001f);

    //2 rows of scores but only 1 answer
    try {
        nnf::cross_entropy(logits, {0});
        fail("cross_entropy with the wrong number of answers should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_cross_entropy_blame() {
    Tensor logits = Tensor::from_vector({2, 1, 0.1f, 0.5f, 2.5f, 0.3f}, {2, 3});
    logits.set_requires_grad(true);
    nnf::cross_entropy(logits, {0, 1}).backward();

    //the blame should be exactly (probabilities − one_hot) ÷ batch size
    Tensor expected = (nnf::softmax(logits) - nnf::one_hot({0, 1}, 3)) * 0.5f;
    for (size_t i = 0; i < 6; ++i) {
        check_number("cross_entropy blame " + std::to_string(i), logits.grad().data()[i], expected.data()[i]);
    }
}

static void test_learns_three_classes() {
    //3 points, each from a different class
    Tensor points = Tensor::from_vector({1, 0, 0, 1, -1, -1}, {3, 2});
    std::vector<size_t> answers = {0, 1, 2};

    nnf::Linear layer(2, 3);  //2 inputs -> 3 scores, one per class
    nnf::SGD opt(layer.parameters(), 0.5f);

    for (int step = 0; step < 100; ++step) {
        opt.zero_grad();
        nnf::cross_entropy(layer.forward(points), answers).backward();
        opt.step();
    }

    //after training, every guess should be the right answer
    std::vector<size_t> guesses = nnf::argmax(layer.forward(points));
    for (size_t i = 0; i < 3; ++i) {
        if (guesses[i] != answers[i])
            fail("point " + std::to_string(i) + " should be class " + std::to_string(answers[i]));
    }
}

int main() {
    test_softmax();
    test_one_hot_and_argmax();
    test_cross_entropy_value();
    test_cross_entropy_blame();
    test_learns_three_classes();
    return report();
}
