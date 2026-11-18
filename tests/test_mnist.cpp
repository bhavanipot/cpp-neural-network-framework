// Tests for reading MNIST files.
// Run with: cmake --build build && ctest --test-dir build --output-on-failure

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "test_helpers.h"  //fail, check_number, check_tensor, report
#include "nnf/mnist.h"

//write a 4-byte number biggest part first, the way MNIST headers store it
static void write_number(std::ofstream& file, size_t n) {
    file.put(static_cast<char>(n / 16777216 % 256));
    file.put(static_cast<char>(n / 65536 % 256));
    file.put(static_cast<char>(n / 256 % 256));
    file.put(static_cast<char>(n % 256));
}

//write a tiny MNIST pair of files: 2 pictures of 2 × 2 pixels, answers 3 and 7
static void write_tiny_mnist(const std::string& images_path, const std::string& labels_path) {
    std::ofstream images(images_path, std::ios::binary);
    write_number(images, 2051);  //"I'm a pictures file"
    write_number(images, 2);     //2 pictures
    write_number(images, 2);     //2 rows
    write_number(images, 2);     //2 columns
    unsigned char pixels[] = {0, 255, 51, 102,   //first picture
                              255, 0, 0, 255};   //second picture
    for (unsigned char p : pixels) {
        images.put(static_cast<char>(p));
    }

    std::ofstream labels(labels_path, std::ios::binary);
    write_number(labels, 2049);  //"I'm an answers file"
    write_number(labels, 2);     //2 answers
    labels.put(3);
    labels.put(7);
}

static void test_load_mnist() {
    write_tiny_mnist("tiny-images", "tiny-labels");
    nnf::MnistData data = nnf::load_mnist("tiny-images", "tiny-labels");

    //pixels scaled from 0-255 to 0-1: 255 -> 1, 51 -> 0.2, 102 -> 0.4
    check_tensor("mnist pictures", data.images, {2, 4}, {0, 1, 0.2f, 0.4f, 1, 0, 0, 1});
    check_number("mnist answers count", data.labels.size(), 2.0f);
    check_number("mnist answer 0", data.labels[0], 3.0f);
    check_number("mnist answer 1", data.labels[1], 7.0f);
}

static void test_batch() {
    write_tiny_mnist("tiny-images", "tiny-labels");
    nnf::MnistData data = nnf::load_mnist("tiny-images", "tiny-labels");

    nnf::MnistData first = nnf::batch(data, 0, 1);
    check_tensor("first batch", first.images, {1, 4}, {0, 1, 0.2f, 0.4f});
    check_number("first batch answer", first.labels[0], 3.0f);

    //asking for 5 from picture 1 only gives the 1 picture that's left
    nnf::MnistData last = nnf::batch(data, 1, 5);
    check_tensor("last batch", last.images, {1, 4}, {1, 0, 0, 1});
    check_number("last batch answer", last.labels[0], 7.0f);
}

static void test_bad_files() {
    //a file that doesn't exist
    try {
        nnf::load_mnist("no-such-file", "no-such-file");
        fail("load_mnist with a missing file should throw");
    } catch (const std::runtime_error&) {}

    //answers passed as pictures: the magic number is 2049, not 2051
    write_tiny_mnist("tiny-images", "tiny-labels");
    try {
        nnf::load_mnist("tiny-labels", "tiny-labels");
        fail("load_mnist with the wrong kind of file should throw");
    } catch (const std::runtime_error&) {}
}

int main() {
    test_load_mnist();
    test_batch();
    test_bad_files();
    return report();
}
