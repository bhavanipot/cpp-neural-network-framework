// Test: train y = 2x + 1 from examples
// Run with: cmake --build build && ctest --test-dir build --output-on-failure

#include <stdexcept>
#include <vector>
#include "test_helpers.h"  //fail, check_number, check_tensor, report
#include "nnf/layers.h"
#include "nnf/optim.h"

static void test_sgd() {
    Tensor w = Tensor::from_vector({3}, {1});
    w.set_requires_grad(true);
    nnf::SGD opt({w}, 0.1f);

    //one step on w × w: blame is 2w = 6, so w becomes 3 - 0.1 × 6 = 2.4
    opt.zero_grad();
    (w * w).backward();
    opt.step();
    check_tensor("one SGD step", w, {1}, {2.4f});

    //zero_grad clears the blame
    opt.zero_grad();
    check_tensor("SGD zero_grad", w.grad(), {1}, {0});

    //an untracked weight is refused
    try {
        nnf::SGD bad({Tensor::zeros({1})}, 0.1f);
        fail("SGD with an untracked weight should throw");
    } catch (const std::invalid_argument&) {}
}

static void test_learns_a_line() {
    //the examples: inputs x and the right answers y = 2x + 1
    Tensor x = Tensor::from_vector({-2, -1, 0, 1, 2, 3}, {6});
    Tensor y = x * 2.0f + 1.0f;

    //start from a far guess
    Tensor w = Tensor::zeros({1});
    Tensor b = Tensor::zeros({1});
    w.set_requires_grad(true);
    b.set_requires_grad(true);
    nnf::SGD opt({w, b}, 0.1f);

    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();

        Tensor diff = x * w + b - y;
        nnf::mean(diff * diff).backward();

        opt.step();
    }

    //after training, w and b should be very close to the real line
    check_number("learned w", w.data()[0], 2.0f, 0.01f);
    check_number("learned b", b.data()[0], 1.0f, 0.01f);
}

//a size × size grid of points from -1.5 to 1.5, labelled 1 if inside the circle of radius 1
//different sizes give different points, e.g. a 23 × 23 grid falls in between a 20 × 20 one
static void make_circle_data(int size, Tensor& points, Tensor& labels) {
    std::vector<float> xy;
    std::vector<float> answers;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float x = -1.5f + 3.0f * i / (size - 1);
            float y = -1.5f + 3.0f * j / (size - 1);
            xy.push_back(x);
            xy.push_back(y);
            answers.push_back(x * x + y * y < 1.0f ? 1.0f : 0.0f);  //inside if x² + y² < 1
        }
    }

    size_t count = static_cast<size_t>(size * size);
    points = Tensor::from_vector(xy, {count, 2});
    labels = Tensor::from_vector(answers, {count, 1});
}

//fraction of points where the guess (score above 0.5 = inside) matches the answer
static float accuracy(const Tensor& scores, const Tensor& labels) {
    size_t correct = 0;
    for (size_t i = 0; i < scores.numel(); ++i) {
        float guess = scores.data()[i] > 0.5f ? 1.0f : 0.0f;
        if (guess == labels.data()[i])
            ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(scores.numel());
}

static void test_learns_a_circle() {
    //learn from a 20 × 20 grid, test on a 23 × 23 grid of points it never saw
    Tensor points = Tensor::zeros({1});
    Tensor labels = Tensor::zeros({1});
    make_circle_data(20, points, labels);
    Tensor test_points = Tensor::zeros({1});
    Tensor test_labels = Tensor::zeros({1});
    make_circle_data(23, test_points, test_labels);

    //plain linear regression can't separate inside from outside
    nnf::Linear line(2, 1);
    nnf::SGD line_opt(line.parameters(), 0.1f);

    //two layers with relu between: 2 inputs -> 16 hidden -> 1 output
    nnf::Linear hidden(2, 16, 1);
    nnf::Linear output(16, 1, 2);
    std::vector<Tensor> params = hidden.parameters();
    for (const Tensor& p : output.parameters()) {
        params.push_back(p);
    }
    nnf::SGD net_opt(params, 0.1f);

    for (int step = 0; step < 300; ++step) {
        line_opt.zero_grad();
        Tensor line_diff = line.forward(points) - labels;
        nnf::mean(line_diff * line_diff).backward();
        line_opt.step();

        net_opt.zero_grad();
        Tensor net_diff = output.forward(nnf::relu(hidden.forward(points))) - labels;
        nnf::mean(net_diff * net_diff).backward();
        net_opt.step();
    }

    //judge both on the test points, which they never learned from
    float line_accuracy = accuracy(line.forward(test_points), test_labels);
    float net_accuracy = accuracy(output.forward(nnf::relu(hidden.forward(test_points))), test_labels);

    if (net_accuracy < 0.9f)
        fail("the network should get over 90% of new circle points right, got " + std::to_string(net_accuracy));
    if (line_accuracy > 0.8f)
        fail("linear regression shouldn't manage the circle, got " + std::to_string(line_accuracy));
}

int main() {
    test_sgd();
    test_learns_a_line();
    test_learns_a_circle();
    return report();
}
