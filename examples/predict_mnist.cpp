// Use the network trained by train_mnist, without training it again.
// Run train_mnist first: it saves the weights to data/mnist_weights.txt

#include <iostream>
#include <stdexcept>
#include <vector>
#include "nnf/tensor.h"
#include "nnf/layers.h"
#include "nnf/losses.h"
#include "nnf/mnist.h"
#include "nnf/save.h"

// the same network as train_mnist.cpp: the saved weights only fit this exact shape
struct MnistNet {
    nnf::Linear hidden{784, 128, 1};
    nnf::Linear output{128, 10, 2};

    nnf::Tensor forward(const nnf::Tensor& x) const {
        return output.forward(nnf::relu(hidden.forward(x)));
    }

    std::vector<nnf::Tensor> parameters() const {
        std::vector<nnf::Tensor> params = hidden.parameters();
        for (const nnf::Tensor& p : output.parameters()) {
            params.push_back(p);
        }
        return params;
    }
};

//draw one 28 × 28 picture: # where there's ink, . where it's blank
static void draw(const nnf::MnistData& data, size_t picture) {
    for (size_t row = 0; row < 28; ++row) {
        for (size_t col = 0; col < 28; ++col) {
            std::cout << (data.images.data()[picture * 784 + row * 28 + col] > 0.5f ? "# " : ". ");
        }
        std::cout << "\n";
    }
}

int main() {
    try {
        nnf::MnistData test = nnf::load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

        //a fresh network starts with random weights...
        MnistNet net;
        //...then gets the trained ones from the file
        nnf::load_parameters(net.parameters(), "data/mnist_weights.txt");

        //count the right guesses on the 10,000 test pictures
        std::vector<size_t> guesses = nnf::argmax(net.forward(test.images));
        size_t correct = 0;
        for (size_t i = 0; i < guesses.size(); ++i) {
            if (guesses[i] == test.labels[i])
                ++correct;
        }
        std::cout << "loaded the trained weights: " << correct << " of " << guesses.size()
                  << " test pictures right\n";

        //show the first 3 it got wrong
        size_t shown = 0;
        for (size_t i = 0; i < guesses.size() && shown < 3; ++i) {
            if (guesses[i] != test.labels[i]) {
                std::cout << "\npicture " << i << ": answer " << test.labels[i] << ", guessed " << guesses[i] << "\n";
                draw(test, i);
                ++shown;
            }
        }
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << " (run scripts/download_mnist.sh, then train_mnist)\n";
    }
}
