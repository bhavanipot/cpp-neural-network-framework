// Small helpers shared by every test file. 

#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "nnf/tensor.h"

using nnf::Tensor;

inline int failures = 0;  //count of how many checks failed

//record a failed check
inline void fail(const std::string& name) {
    ++failures;
    std::cout << "FAIL: " << name << "\n";
}

//check two numbers are equal, to within a small tolerance (0.0001 unless given)
inline void check_number(const std::string& name, float got, float expected, float tolerance = 1e-4f) {
    if (std::fabs(got - expected) > tolerance)
        fail(name + ", got " + std::to_string(got));
}

//check a tensor has the expected shape and numbers
inline void check_tensor(const std::string& name, const Tensor& t,
                         const nnf::Shape& shape, const std::vector<float>& expected) {
    if (t.shape() != shape) {
        fail(name + ", wrong shape: " + t.to_string());
        return;
    }

    Tensor ordered = t.contiguous();  //so transposed tensors can be checked too
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(ordered.data()[i] - expected[i]) > 1e-4f) {
            fail(name + ", got " + t.to_string());
            return;
        }
    }
}

//print the summary line and give the exit code (non-zero tells ctest something failed)
inline int report() {
    if (failures == 0)
        std::cout << "all tests passed\n";
    else
        std::cout << failures << " check(s) failed\n";

    return failures == 0 ? 0 : 1;
}
