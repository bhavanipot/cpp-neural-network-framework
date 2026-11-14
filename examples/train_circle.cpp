// Teach the framework to tell points inside a circle from points outside it.
// A straight line can't do this, so we compare plain linear regression with a
// small network: Linear(2, 16) -> relu -> Linear(16, 1).

#include <iostream>
#include <vector>
#include "nnf/tensor.h"
#include "nnf/layers.h"
#include "nnf/optim.h"

//a size × size grid of points from -1.5 to 1.5, labelled 1 if inside the circle of radius 1
//different sizes give different points, e.g. a 23 × 23 grid falls in between a 20 × 20 one
static void make_circle_data(int size, nnf::Tensor& points, nnf::Tensor& labels) {
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
    points = nnf::Tensor::from_vector(xy, {count, 2});       //one row of (x, y) per point
    labels = nnf::Tensor::from_vector(answers, {count, 1});  //one answer per point
}

// two layers with relu between: 2 inputs -> 16 hidden -> 1 output
struct CircleNet {
    nnf::Linear hidden{2, 16, 1};  //16 neurons, each learns one straight edge
    nnf::Linear output{16, 1, 2};  //combines the edges into one score

    nnf::Tensor forward(const nnf::Tensor& x) const {
        return output.forward(nnf::relu(hidden.forward(x)));
    }

    //every weight from both layers, for the optimizer
    std::vector<nnf::Tensor> parameters() const {
        std::vector<nnf::Tensor> params = hidden.parameters();
        for (const nnf::Tensor& p : output.parameters()) {
            params.push_back(p);
        }
        return params;
    }
};

//fraction of points where the guess (score above 0.5 = inside) matches the answer
static float accuracy(const nnf::Tensor& scores, const nnf::Tensor& labels) {
    size_t correct = 0;
    for (size_t i = 0; i < scores.numel(); ++i) {
        float guess = scores.data()[i] > 0.5f ? 1.0f : 0.0f;
        if (guess == labels.data()[i])
            ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(scores.numel());
}

//draw a size × size grid as a picture: # = inside (score above 0.5), . = outside
//point (i, j) is stored at row i * size + j, where i steps along x and j along y
static void draw(const std::string& title, const nnf::Tensor& scores, int size) {
    std::cout << title << "\n";
    for (int j = size - 1; j >= 0; --j) {  //top row is the biggest y
        for (int i = 0; i < size; ++i) {
            std::cout << (scores.data()[i * size + j] > 0.5f ? "# " : ". ");
        }
        std::cout << "\n";
    }
}

int main() {
    //training set: a 20 × 20 grid the models learn from
    nnf::Tensor points = nnf::Tensor::zeros({1});
    nnf::Tensor labels = nnf::Tensor::zeros({1});
    make_circle_data(20, points, labels);

    //test set: a 23 × 23 grid of points in between, never used for learning
    nnf::Tensor test_points = nnf::Tensor::zeros({1});
    nnf::Tensor test_labels = nnf::Tensor::zeros({1});
    make_circle_data(23, test_points, test_labels);

    //the two models: plain linear regression, and the network
    nnf::Linear line(2, 1);
    CircleNet net;
    nnf::SGD line_opt(line.parameters(), 0.1f);
    nnf::SGD net_opt(net.parameters(), 0.1f);

    for (int step = 0; step <= 2000; ++step) {
        //train linear regression one step
        line_opt.zero_grad();
        nnf::Tensor line_diff = line.forward(points) - labels;
        nnf::Tensor line_loss = nnf::mean(line_diff * line_diff);
        line_loss.backward();
        line_opt.step();

        //train the network one step
        net_opt.zero_grad();
        nnf::Tensor net_diff = net.forward(points) - labels;
        nnf::Tensor net_loss = nnf::mean(net_diff * net_diff);
        net_loss.backward();
        net_opt.step();

        if (step % 200 == 0)
            std::cout << "step " << step
                      << ": linear loss " << line_loss.sum() << ", accuracy " << accuracy(line.forward(points), labels)
                      << " | network loss " << net_loss.sum() << ", accuracy " << accuracy(net.forward(points), labels) << "\n";
    }

    //the real check: accuracy on points the models never learned from
    std::cout << "\n";
    std::cout << "linear regression: training accuracy " << accuracy(line.forward(points), labels)
              << ", test accuracy " << accuracy(line.forward(test_points), test_labels) << "\n";
    std::cout << "network:           training accuracy " << accuracy(net.forward(points), labels)
              << ", test accuracy " << accuracy(net.forward(test_points), test_labels) << "\n";

    std::cout << "\n";
    draw("the real answer (test set):", test_labels, 23);
    std::cout << "\n";
    draw("the network's guess (test set):", net.forward(test_points), 23);
}
