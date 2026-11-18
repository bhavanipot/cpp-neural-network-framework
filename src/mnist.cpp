#include <algorithm>   //std::min
#include <fstream>     //std::ifstream
#include <stdexcept>   //std::runtime_error
#include "nnf/mnist.h"

//mnist is a dataset of handwritten digits, used for training and testing neural networks

namespace nnf {

//read a 4-byte number stored biggest part first, as the MNIST headers are
//bytes 0, 0, 39, 16 -> 39 × 256 + 16 = 10000
static size_t read_number(std::ifstream& file) {
    unsigned char bytes[4] = {0, 0, 0, 0};
    file.read(reinterpret_cast<char*>(bytes), 4);
    return bytes[0] * 16777216u + bytes[1] * 65536u + bytes[2] * 256u + bytes[3];
}

// read an MNIST pictures file and its answers file
MnistData load_mnist(const std::string& images_path, const std::string& labels_path) {
    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!images_file || !labels_file)
        throw std::runtime_error("load_mnist: can't open " + images_path + " or " + labels_path);

    //pictures header: 2051, how many, rows, columns
    if (read_number(images_file) != 2051)
        throw std::runtime_error("load_mnist: " + images_path + " is not an MNIST pictures file");
    size_t count = read_number(images_file);
    size_t rows = read_number(images_file);
    size_t cols = read_number(images_file);

    //answers header: 2049, how many (must match the pictures)
    if (read_number(labels_file) != 2049)
        throw std::runtime_error("load_mnist: " + labels_path + " is not an MNIST answers file");
    if (read_number(labels_file) != count)
        throw std::runtime_error("load_mnist: different number of pictures and answers");

    //the rest of each file: one byte per pixel, one byte per answer
    std::vector<unsigned char> pixels(count * rows * cols);
    std::vector<unsigned char> answers(count);
    images_file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    labels_file.read(reinterpret_cast<char*>(answers.data()), static_cast<std::streamsize>(answers.size()));
    if (!images_file || !labels_file)
        throw std::runtime_error("load_mnist: file is shorter than its header says");

    //pixels from 0-255 down to 0-1, so the network starts with small numbers
    std::vector<float> scaled(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        scaled[i] = pixels[i] / 255.0f;
    }

    std::vector<size_t> labels(count);
    for (size_t i = 0; i < count; ++i) {
        labels[i] = answers[i];
    }

    return MnistData{Tensor::from_vector(scaled, {count, rows * cols}), labels};
}

// pictures start, start + 1, ... as a smaller set (the last batch may be shorter)
//batch(train, 128, 64) = pictures 128 to 191
MnistData batch(const MnistData& data, size_t start, size_t count) {
    size_t pixels = data.images.shape()[1];
    size_t end = std::min(start + count, data.labels.size());

    std::vector<float> images;
    std::vector<size_t> labels;
    for (size_t row = start; row < end; ++row) {
        for (size_t p = 0; p < pixels; ++p) {
            images.push_back(data.images.data()[row * pixels + p]);
        }
        labels.push_back(data.labels[row]);
    }

    return MnistData{Tensor::from_vector(images, {end - start, pixels}), labels};
}

}  // namespace nnf
