#pragma once
#include <vector>
#include "nnf/tensor.h"

namespace nnf {

// helpers for choosing between several answers (classes)
//scores ("logits") have shape {batch, classes}: one row of scores per example

Tensor softmax(const Tensor& logits);      //each row into probabilities that add up to 1
Tensor log_softmax(const Tensor& logits);  //log of softmax, calculated safely

Tensor one_hot(const std::vector<size_t>& targets, size_t classes);  //answer 2 of 3 -> [0, 0, 1]

//the average of −log(probability given to the right answer), as a single tracked number
Tensor cross_entropy(const Tensor& logits, const std::vector<size_t>& targets);

std::vector<size_t> argmax(const Tensor& logits);  //the highest-scoring class in each row: the guess

}  // namespace nnf
