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

    // Ask for a position that doesn't exist
    try {
        t.at({5, 0});
    } catch (const std::out_of_range& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }
}
