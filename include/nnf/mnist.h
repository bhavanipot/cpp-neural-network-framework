#pragma once
#include <string>
#include <vector>
#include "nnf/tensor.h"

namespace nnf {

// a set of digit pictures and their right answers
struct MnistData {
    Tensor images;               //{count, 784}: one row of 28 × 28 pixels per picture, from 0 to 1
    std::vector<size_t> labels;  //the digit (0 to 9) each picture shows
};

//read an MNIST pictures file and its answers file (download them with scripts/download_mnist.sh)
MnistData load_mnist(const std::string& images_path, const std::string& labels_path);

//pictures start, start + 1, ... as a smaller set, for training a few at a time
MnistData batch(const MnistData& data, size_t start, size_t count);

}  // namespace nnf
