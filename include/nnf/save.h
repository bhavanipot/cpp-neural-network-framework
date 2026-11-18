#pragma once
#include <string>
#include <vector>
#include "nnf/tensor.h"

namespace nnf {

// keep what a network learned after the program ends
//usage: save_parameters(net.parameters(), path) after training,
//       load_parameters(fresh_net.parameters(), path) to use it again without training

//write every tensor's shape and numbers to a text file
void save_parameters(const std::vector<Tensor>& params, const std::string& path);

//read them back into tensors of the same shapes, e.g. a fresh network's parameters
void load_parameters(const std::vector<Tensor>& params, const std::string& path);

}  // namespace nnf
