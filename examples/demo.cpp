#include <iostream>
#include "nnf/tensor.h"

int main() {
    //Test program for tensor class
  
    nnf::Tensor t = nnf::Tensor::zeros({2, 3});  // 2x3 tensor full of zeros
    std::cout << "numel: " << t.numel() << "\n"; // Return the amount fo  numbers does it hold. 
    t.at({1, 2}) = 5.0f; // Change row 1, column 2 to 5

    // Read some numbers back
    std::cout << "t[1][2]: " << t.at({1, 2}) << "\n";     // 5
    std::cout << "t[0][0]: " << t.at({0, 0}) << "\n";     // 0

    // Make a tensor full of 7s
    nnf::Tensor sevens = nnf::Tensor::full({2, 2}, 7.0f);
    std::cout << "sevens[1][1]: " << sevens.at({1, 1}) << "\n";  // 7

    //Test Program for the to_string() method

    // Creating a sample tester
    nnf::Tensor m = nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    std::cout << "m[0][1]: " << m.at({0, 1}) << "\n";   // 2
    std::cout << "m[1][0]: " << m.at({1, 0}) << "\n";

    std::cout << "--- to_string tests ---\n";
    std::cout << m.to_string() << "\n";
    std::cout << sevens.to_string() << "\n";     // a tensor full of one value
    std::cout << nnf::Tensor::zeros({3}).to_string() << "\n";
    std::cout << nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}).to_string() << "\n";  // a 3D tensor (2 tables of 2x2)
    std::cout << nnf::Tensor::full({}, 5.0f).to_string() << "\n";

    m.at({0, 0}) = 100.0f;
    std::cout << m.to_string() << "\n";

    // Ask for a position that doesn't exist
    try {
        t.at({5, 0});
    } catch (const std::out_of_range& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }

    try {
        nnf::Tensor::from_vector({1, 2, 3}, {2, 2});
    } catch (const std::invalid_argument& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }

}

