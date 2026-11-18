#include <fstream>     //std::ofstream, std::ifstream
#include <iomanip>     //std::setprecision
#include <stdexcept>   //std::runtime_error
#include "nnf/save.h"

namespace nnf {

// write every tensor to a text file: its shape on one line, then its numbers on the next
//a {2, 3} tensor becomes:
//2 2 3                       <- 2 sizes: 2 and 3
//0.5 -1.25 0 3 0.75 -2       <- its 6 numbers in reading order
void save_parameters(const std::vector<Tensor>& params, const std::string& path) {
    std::ofstream file(path);
    if (!file)
        throw std::runtime_error("save_parameters: can't write " + path);

    file << std::setprecision(9);  //enough digits that every float comes back exactly

    for (const Tensor& p : params) {
        //how many sizes, then the sizes: {784, 128} -> "2 784 128"
        file << p.shape().size();
        for (size_t size : p.shape()) {
            file << " " << size;
        }
        file << "\n";

        Tensor ordered = p.contiguous();
        for (size_t i = 0; i < p.numel(); ++i) {
            file << ordered.data()[i] << " ";
        }
        file << "\n";
    }
}

// read the numbers back, in the same order they were saved
//writes straight into the tensors, like SGD::step, so no receipt is written
void load_parameters(const std::vector<Tensor>& params, const std::string& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("load_parameters: can't open " + path);

    for (Tensor p : params) {  //a copy of the handle still points to the same numbers
        size_t dims = 0;
        file >> dims;
        Shape shape(dims);
        for (size_t d = 0; d < dims; ++d) {
            file >> shape[d];
        }

        //the file must hold weights of exactly this shape
        if (!file || shape != p.shape())
            throw std::runtime_error("load_parameters: " + path + " doesn't match these weights' shapes");

        for (size_t i = 0; i < p.numel(); ++i) {
            file >> p.data()[i];
        }
        if (!file)
            throw std::runtime_error("load_parameters: " + path + " ends too early");
    }
}

}  // namespace nnf
