#include <algorithm>   //std::max
#include <stdexcept>   //std::invalid_argument
#include "nnf/losses.h"

namespace nnf {

//the biggest score in each row, as a {batch, 1} column
//not tracked: subtracting it doesn't change softmax, so it needs no blame
static Tensor row_max(const Tensor& logits) {
    Tensor ordered = logits.contiguous();
    size_t rows = logits.shape()[0];
    size_t cols = logits.shape()[1];

    std::vector<float> maxes(rows);
    for (size_t row = 0; row < rows; ++row) {
        maxes[row] = ordered.data()[row * cols];
        for (size_t col = 1; col < cols; ++col) {
            maxes[row] = std::max(maxes[row], ordered.data()[row * cols + col]);
        }
    }

    return Tensor::from_vector(maxes, {rows, 1});
}

// each row into probabilities: exp, then divide by the row's total
//[2, 1, 0.1] -> [0.659, 0.242, 0.099]
Tensor softmax(const Tensor& logits) {
    size_t rows = logits.shape()[0];

    //subtract each row's biggest score first, so exp can't overflow
    Tensor e = exp(logits - row_max(logits));
    return e / e.sum(1).reshape({rows, 1});
}

// log of softmax, rearranged so it never takes the log of a tiny number:
//log_softmax = (scores - max) - log(sum(exp(scores - max)))
Tensor log_softmax(const Tensor& logits) {
    size_t rows = logits.shape()[0];

    Tensor shifted = logits - row_max(logits);
    return shifted - log(exp(shifted).sum(1)).reshape({rows, 1});
}

// the right answers as rows: 1 at the correct class, 0 elsewhere
//one_hot({2, 0}, 3) = [[0, 0, 1], [1, 0, 0]]
Tensor one_hot(const std::vector<size_t>& targets, size_t classes) {
    Tensor result = Tensor::zeros({targets.size(), classes});

    for (size_t row = 0; row < targets.size(); ++row) {
        //every answer must be one of the classes
        if (targets[row] >= classes)
            throw std::invalid_argument("one_hot: answer is not one of the classes");
        result.data()[row * classes + targets[row]] = 1.0f;
    }

    return result;
}

// how surprised the network was by the right answers, averaged over the batch:
//mean of −log(probability given to the right answer)
Tensor cross_entropy(const Tensor& logits, const std::vector<size_t>& targets) {
    size_t rows = logits.shape()[0];
    size_t classes = logits.shape()[1];

    //one answer per example
    if (targets.size() != rows)
        throw std::invalid_argument("cross_entropy: need one answer per row of scores");

    //one_hot × log_softmax keeps only the log probability of each right answer
    Tensor picked = one_hot(targets, classes) * log_softmax(logits);
    return sum(picked) * (-1.0f / static_cast<float>(rows));
}

// the highest-scoring class in each row: the network's guess
//[[2, 1, 0.1], [0.5, 2.5, 0.3]] -> {0, 1}
std::vector<size_t> argmax(const Tensor& logits) {
    Tensor ordered = logits.contiguous();
    size_t rows = logits.shape()[0];
    size_t cols = logits.shape()[1];

    std::vector<size_t> guesses(rows);
    for (size_t row = 0; row < rows; ++row) {
        size_t best = 0;
        for (size_t col = 1; col < cols; ++col) {
            if (ordered.data()[row * cols + col] > ordered.data()[row * cols + best])
                best = col;
        }
        guesses[row] = best;
    }

    return guesses;
}

}  // namespace nnf
