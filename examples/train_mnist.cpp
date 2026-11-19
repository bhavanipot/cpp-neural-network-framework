// Teach the framework to read handwritten digits (MNIST).
// Download the data first: sh scripts/download_mnist.sh
// Build in Release so it runs fast:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/train_mnist

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "nnf/tensor.h"
#include "nnf/layers.h"
#include "nnf/losses.h"
#include "nnf/mnist.h"
#include "nnf/optim.h"
#include "nnf/save.h"

// two layers with relu between: 784 pixels -> 128 hidden -> 10 scores, one per digit
struct MnistNet {
    nnf::Linear hidden{784, 128, 1};  //128 neurons, each learns to spot a stroke or shape
    nnf::Linear output{128, 10, 2};   //combines them into a score for each digit

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

//fraction of pictures where the guess (the highest score) is the right digit
static float accuracy(const MnistNet& net, const nnf::MnistData& data) {
    std::vector<size_t> guesses = nnf::argmax(net.forward(data.images));

    size_t correct = 0;
    for (size_t i = 0; i < guesses.size(); ++i) {
        if (guesses[i] == data.labels[i])
            ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(guesses.size());
}

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
        //training set: 60,000 pictures to learn from
        //test set: 10,000 pictures never used for learning
        nnf::MnistData train = nnf::load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
        nnf::MnistData test = nnf::load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

        MnistNet net;
        nnf::SGD opt(net.parameters(), 0.1f);
        size_t batch_size = 64;

        //before training: guessing at random, so about 1 in 10 right
        std::cout << "before training: test accuracy " << accuracy(net, test) << "\n";

        for (int epoch = 1; epoch <= 3; ++epoch) {
            //time each epoch, to see how fast training is
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

            //one epoch = one pass through all 60,000 training pictures, 64 at a time
            for (size_t start = 0; start < train.labels.size(); start += batch_size) {
                nnf::MnistData b = nnf::batch(train, start, batch_size);

                opt.zero_grad();
                nnf::Tensor loss = nnf::cross_entropy(net.forward(b.images), b.labels);
                loss.backward();
                opt.step();

                if (start % 12800 == 0)
                    std::cout << "epoch " << epoch << ", picture " << start << ": loss " << loss.sum() << "\n";
            }

            std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;

            //after each pass, check on the 10,000 pictures it never learns from
            std::cout << "after epoch " << epoch << ": test accuracy " << accuracy(net, test)
                      << " (" << seconds.count() << " seconds)\n";
        }

        //keep what it learned, so predict_mnist can use it without training again
        nnf::save_parameters(net.parameters(), "data/mnist_weights.txt");
        std::cout << "saved the weights to data/mnist_weights.txt\n";

        //show the first few test pictures with the network's guess
        nnf::MnistData few = nnf::batch(test, 0, 3);
        std::vector<size_t> guesses = nnf::argmax(net.forward(few.images));
        for (size_t i = 0; i < 3; ++i) {
            std::cout << "\nanswer " << few.labels[i] << ", guessed " << guesses[i] << "\n";
            draw(few, i);
        }
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << " (run scripts/download_mnist.sh first)\n";
    }
}
