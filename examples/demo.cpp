#include <iostream>
#include "nnf/tensor.h"

int main() {
    //Test program for tensor class
  
    nnf::Tensor t = nnf::Tensor::zeros({2, 3});  // 2x3 tensor full of zeros
    std::cout << "numel: " << t.numel() << "\n"; // Return the amount fo  numbers does it hold. 
    t.at({1, 2}) = 5.0f; // Change row 1, column 2 to 5

    // Read numbers
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

    // ---- randn ----
    std::cout << "--- randn tests ---\n";
    std::cout << nnf::Tensor::randn({2, 3}).to_string() << "\n";
    std::cout << nnf::Tensor::randn({2, 3}).to_string() << "\n";      
    std::cout << nnf::Tensor::randn({2, 3}, 7).to_string() << "\n";
    
    // ---- math ----
    std::cout << "--- math tests ---\n";
    nnf::Tensor a = nnf::Tensor::from_vector({1, 2, 3}, {3});
    nnf::Tensor b = nnf::Tensor::from_vector({4, 5, 6}, {3});
    std::cout << (a + b).to_string() << "\n";   // [5, 7, 9]
    std::cout << (a - b).to_string() << "\n";   // [-3, -3, -3]
    std::cout << (a * b).to_string() << "\n";   // [4, 10, 18]
    std::cout << (a / b).to_string() << "\n";   // [0.25, 0.4, 0.5]
    std::cout << (-a).to_string() << "\n";      // [-1, -2, -3]
    std::cout << (+a).to_string() << "\n";      // [1, 2, 3]

    // ---- scalar math ----
    std::cout << "--- scalar math tests ---\n";
    std::cout << (a + 10.0f).to_string() << "\n";  // [11, 12, 13]
    std::cout << (a - 10.0f).to_string() << "\n";  // [-9, -8, -7]
    std::cout << (a * 10.0f).to_string() << "\n";  // [10, 20, 30]
    std::cout << (a / 10.0f).to_string() << "\n";  // [0.1, 0.2, 0.3]
    std::cout << (10.0f + a).to_string() << "\n";  // [11, 12, 13]
    std::cout << (10.0f - a).to_string() << "\n";  // [9, 8, 7]
    std::cout << (10.0f * a).to_string() << "\n";  // [10, 20, 30]
    std::cout << (10.0f / a).to_string() << "\n";  // [10, 5, 3.33333]

    // ---- reductions ----
    std::cout << "--- reduction tests ---\n";
    std::cout << "a.sum(): " << a.sum() << "\n";    // 6  (1+2+3)
    std::cout << "a.mean(): " << a.mean() << "\n";  // 2  (6/3)

    nnf::Tensor grid = nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    std::cout << "grid.sum(): " << grid.sum() << "\n";    // 21
    std::cout << "grid.mean(): " << grid.mean() << "\n";  // 3.5

    try {
        nnf::Tensor::zeros({0}).mean();
    } catch (const std::invalid_argument& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }

    // ---- matmul ----
    std::cout << "--- matmul tests ---\n";
    nnf::Tensor x = nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});   // 2x3
    nnf::Tensor y = nnf::Tensor::from_vector({7, 8, 9, 10, 11, 12}, {3, 2}); // 3x2
    std::cout << nnf::matmul(x, y).to_string() << "\n";  // {2,2}: [58, 64, 139, 154]

    try {
        nnf::matmul(x, x);  // {2,3} x {2,3}: inner dimensions don't match
    } catch (const std::invalid_argument& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }

    // ---- reshape ----
    std::cout << "--- reshape tests ---\n";
    nnf::Tensor flat = nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6}, {6});
    nnf::Tensor grid2 = flat.reshape({2, 3});
    std::cout << grid2.to_string() << "\n";  // same 6 numbers, now shape {2,3}

    // reshape shares storage
    grid2.at({0, 0}) = 99.0f;
    std::cout << flat.to_string() << "\n";   // flat[0] is now 99 too

    try {
        flat.reshape({2, 2});  // 6 elements don't fit into 4
    } catch (const std::invalid_argument& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }

    // ---- transpose ----
    std::cout << "--- transpose tests ---\n";
    nnf::Tensor mat = nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    nnf::Tensor mat_t = mat.transpose();
    std::cout << mat_t.to_string() << "\n";  // {3,2}: [1, 4, 2, 5, 3, 6]
    std::cout << "mat_t[0][1]: " << mat_t.at({0, 1}) << "\n";  // 4
    std::cout << "mat contiguous: " << mat.is_contiguous() << "\n";      // 1 (true)
    std::cout << "mat_t contiguous: " << mat_t.is_contiguous() << "\n";  // 0 (false)

    // transpose shares storage, so editing mat_t also changes mat
    mat_t.at({2, 0}) = 30.0f;               // mat_t[2][0] is mat[0][2]
    std::cout << mat.to_string() << "\n";   // [1, 2, 30, 4, 5, 6]

    // + - * / put a transposed tensor in order first, so this works
    std::cout << (mat_t + mat_t).to_string() << "\n";  // [2, 8, 4, 10, 60, 12]

    // ---- contiguous ----
    std::cout << "--- contiguous tests ---\n";
    nnf::Tensor mat_tc = mat_t.contiguous();
    std::cout << mat_tc.to_string() << "\n";  // [1, 4, 2, 5, 30, 6], same as mat_t
    std::cout << "mat_tc contiguous: " << mat_tc.is_contiguous() << "\n";  // 1 (true)
    std::cout << (mat_tc + mat_tc).to_string() << "\n";  // math works now: [2, 8, 4, 10, 60, 12]
    std::cout << nnf::matmul(mat, mat_t.contiguous()).to_string() << "\n";  // {2,3} x {3,2} -> {2,2}: [905, 194, 194, 77]

    // ---- elementwise functions ----
    std::cout << "--- elementwise function tests ---\n";
    nnf::Tensor v = nnf::Tensor::from_vector({-2, 0, 1, 2}, {4});
    std::cout << nnf::relu(v).to_string() << "\n";  // [0, 0, 1, 2]
    std::cout << nnf::exp(v).to_string() << "\n";   // [0.135, 1, 2.718, 7.389]
    std::cout << nnf::tanh(v).to_string() << "\n";  // [-0.964, 0, 0.762, 0.964]
    std::cout << nnf::log(nnf::exp(v)).to_string() << "\n";  // [-2, 0, 1, 2], log undoes exp

    // ---- sum and mean along an axis ----
    std::cout << "--- axis reduction tests ---\n";
    std::cout << grid.sum(0).to_string() << "\n";    // {3}: [5, 7, 9]
    std::cout << grid.sum(1).to_string() << "\n";    // {2}: [6, 15]
    std::cout << grid.mean(0).to_string() << "\n";   // {3}: [2.5, 3.5, 4.5]
    std::cout << grid.mean(1).to_string() << "\n";   // {2}: [2, 5]

    try {
        grid.sum(2);  // grid only has axes 0 and 1
    } catch (const std::invalid_argument& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }

    // ---- broadcasting ----
    std::cout << "--- broadcasting tests ---\n";
    nnf::Tensor bias = nnf::Tensor::from_vector({10, 20, 30}, {3});
    std::cout << (grid + bias).to_string() << "\n";  // {2,3}: [11, 22, 33, 14, 25, 36]
    std::cout << (bias + grid).to_string() << "\n";  // same, order doesn't matter

    nnf::Tensor totals = grid.sum(1).reshape({2, 1});  // {2} -> {2,1}, a column
    nnf::Tensor probs = grid / totals;                 // each row divided by its own total
    std::cout << probs.to_string() << "\n";            // [0.167, 0.333, 0.5, 0.267, 0.333, 0.4]
    std::cout << probs.sum(1).to_string() << "\n";     // [1, 1], each row adds up to 1

    try {
        grid + nnf::Tensor::from_vector({1, 2}, {2});  // 3 and 2 don't fit
    } catch (const std::invalid_argument& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }

    // ---- autograd: tracking ----
    std::cout << "--- autograd tracking tests ---\n";
    nnf::Tensor w = nnf::Tensor::from_vector({1, 2, 3}, {3});
    std::cout << "w tracked: " << w.requires_grad() << "\n";  // 0 (false)
    w.set_requires_grad(true);
    std::cout << "w tracked: " << w.requires_grad() << "\n";  // 1 (true)
    std::cout << w.grad().to_string() << "\n";                // [0, 0, 0]

    try {
        w.grad();  // w was untracked after set_requires_grad(false), so it has no grad
    } catch (const std::invalid_argument& e) {
        std::cout << "caught error: " << e.what() << "\n";
    }

    // ---- autograd: the receipt ----
    std::cout << "--- autograd receipt tests ---\n";
    nnf::Tensor p = nnf::Tensor::from_vector({3}, {1});
    nnf::Tensor q = nnf::Tensor::from_vector({4}, {1});
    p.set_requires_grad(true);
    q.set_requires_grad(true);

    nnf::Tensor prod = p * q;
    std::cout << prod.to_string() << "\n";                           // [12]
    std::cout << "prod made by: " << nnf::op_name(prod.op()) << "\n";  // Mul
    std::cout << "prod parents: " << prod.num_parents() << "\n";     // 2
    std::cout << prod.parent(0).to_string() << "\n";                 // [3], that's p
    std::cout << prod.parent(1).to_string() << "\n";                 // [4], that's q

    nnf::Tensor chain = prod + p;  // the receipt grows: chain <- prod <- p, q
    std::cout << "chain made by: " << nnf::op_name(chain.op()) << "\n";  // Add
    std::cout << chain.parent(0).to_string() << "\n";                  // [12], that's prod

    std::cout << "a + b made by: " << nnf::op_name((a + b).op()) << "\n";  // None, untracked

    // ---- autograd: backward ----
    std::cout << "--- autograd backward tests ---\n";
    chain.backward();  // chain = p * q + p = 15, read the receipt backwards
    std::cout << "p.grad: " << p.grad().to_string() << "\n";       // [5], q + 1, since p was used twice
    std::cout << "q.grad: " << q.grad().to_string() << "\n";       // [3], p
    std::cout << "prod.grad: " << prod.grad().to_string() << "\n"; // [1], prod was added straight into chain

    // ---- autograd: checking blame the slow way ----
    std::cout << "--- gradient check tests ---\n";
    nnf::Tensor p_up = nnf::Tensor::from_vector({3.01f}, {1});    // p nudged up
    nnf::Tensor p_down = nnf::Tensor::from_vector({2.99f}, {1});  // p nudged down
    nnf::Tensor q_plain = nnf::Tensor::from_vector({4}, {1});
    float up = (p_up * q_plain + p_up).sum();        // 15.0501
    float down = (p_down * q_plain + p_down).sum();  // 14.9501
    std::cout << "nudging p gives: " << (up - down) / 0.02f << "\n";  // 5, same as p.grad

    // ---- autograd: sum and mean ----
    std::cout << "--- autograd sum and mean tests ---\n";
    nnf::Tensor wv = nnf::Tensor::from_vector({1, 2, 3}, {3});  // weights
    nnf::Tensor xv = nnf::Tensor::from_vector({4, 5, 6}, {3});  // input data
    wv.set_requires_grad(true);

    nnf::Tensor out = nnf::sum(wv * xv);  // a single neuron: 1×4 + 2×5 + 3×6 = [32]
    std::cout << out.to_string() << "\n";
    out.backward();
    std::cout << "wv.grad: " << wv.grad().to_string() << "\n";  // [4, 5, 6], each weight's blame is its input

    nnf::Tensor mv = nnf::Tensor::from_vector({1, 2, 3}, {3});
    mv.set_requires_grad(true);
    nnf::mean(mv).backward();                                   // mean([1,2,3]) = [2]
    std::cout << "mv.grad: " << mv.grad().to_string() << "\n";  // [0.333, 0.333, 0.333], each counts 1/3

    // ---- autograd: subtraction, division and scalars ----
    std::cout << "--- autograd arithmetic tests ---\n";
    nnf::Tensor pred = nnf::Tensor::from_vector({2.5f, 0, 2}, {3});     // the network's guesses
    nnf::Tensor target = nnf::Tensor::from_vector({3, -0.5f, 2}, {3});  // the right answers
    pred.set_requires_grad(true);

    nnf::Tensor diff = pred - target;           // [-0.5, 0.5, 0]
    nnf::Tensor loss = nnf::mean(diff * diff);  // squared error: (0.25 + 0.25 + 0) / 3 = [0.1667]
    std::cout << "loss: " << loss.to_string() << "\n";
    loss.backward();
    std::cout << "pred.grad: " << pred.grad().to_string() << "\n";  // [-0.333, 0.333, 0]: up, down, leave alone

    nnf::Tensor s = nnf::Tensor::from_vector({2}, {1});
    s.set_requires_grad(true);
    (10.0f / s - s * 3.0f).backward();                        // 10/2 - 2*3 = -1
    std::cout << "s.grad: " << s.grad().to_string() << "\n";  // [-5.5], -10/2² - 3

    // ---- autograd: activation functions ----
    std::cout << "--- autograd activation tests ---\n";
    nnf::Tensor h = nnf::Tensor::from_vector({-1, 0.5f, 2}, {3});
    h.set_requires_grad(true);
    nnf::sum(nnf::relu(h)).backward();
    std::cout << "relu blame: " << h.grad().to_string() << "\n";  // [0, 1, 1], the negative is blocked

    nnf::Tensor e = nnf::Tensor::from_vector({0, 1}, {2});
    e.set_requires_grad(true);
    nnf::sum(nnf::exp(e)).backward();
    std::cout << "exp blame: " << e.grad().to_string() << "\n";   // [1, 2.718], same as exp(e)

    nnf::Tensor l = nnf::Tensor::from_vector({1, 4}, {2});
    l.set_requires_grad(true);
    nnf::sum(nnf::log(l)).backward();
    std::cout << "log blame: " << l.grad().to_string() << "\n";   // [1, 0.25], 1/x

    nnf::Tensor th = nnf::Tensor::from_vector({0, 2}, {2});
    th.set_requires_grad(true);
    nnf::sum(nnf::tanh(th)).backward();
    std::cout << "tanh blame: " << th.grad().to_string() << "\n"; // [1, 0.071], flat once it squashes

    // ---- autograd: matmul and transpose ----
    std::cout << "--- autograd matmul tests ---\n";
    nnf::Tensor xin = nnf::Tensor::from_vector({1, 2}, {1, 2});                    // one example, 2 features
    nnf::Tensor W = nnf::Tensor::from_vector({0.5f, -1, 2, 0.25f, 1, 3}, {2, 3});  // 2 inputs -> 3 outputs
    W.set_requires_grad(true);

    nnf::Tensor layer_out = nnf::matmul(xin, W);  // a tiny layer: [1, 1, 8]
    std::cout << layer_out.to_string() << "\n";
    nnf::sum(layer_out).backward();
    std::cout << "W.grad: " << W.grad().to_string() << "\n";  // [1, 1, 1, 2, 2, 2], each weight's blame is its input feature

    nnf::Tensor tm = nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    tm.set_requires_grad(true);
    nnf::Tensor wts = nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    nnf::sum(tm.transpose() * wts).backward();
    std::cout << "tm.grad: " << tm.grad().to_string() << "\n";  // [1, 3, 5, 2, 4, 6], wts flipped back to tm's shape

    // ---- autograd: broadcasting ----
    std::cout << "--- autograd broadcasting tests ---\n";
    nnf::Tensor xb = nnf::Tensor::from_vector({1, 2, 3, 4}, {2, 2});  // 2 examples, 2 features
    nnf::Tensor Wb = nnf::Tensor::from_vector({1, 0, 0, 1}, {2, 2});  // weights
    nnf::Tensor bb = nnf::Tensor::from_vector({0.5f, -0.5f}, {2});    // bias, repeated for both examples
    Wb.set_requires_grad(true);
    bb.set_requires_grad(true);

    nnf::sum(nnf::matmul(xb, Wb) + bb).backward();
    std::cout << "bb.grad: " << bb.grad().to_string() << "\n";  // [2, 2], used by both examples, so blame from both
    std::cout << "Wb.grad: " << Wb.grad().to_string() << "\n";  // [4, 4, 6, 6]

    nnf::Tensor gr = nnf::Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    gr.set_requires_grad(true);
    nnf::sum(gr.sum(1) * nnf::Tensor::from_vector({1, 10}, {2})).backward();  // row totals, weighted
    std::cout << "gr.grad: " << gr.grad().to_string() << "\n";  // [1, 1, 1, 10, 10, 10], each row gets its total's blame

    // ---- autograd: zero_grad ----
    std::cout << "--- autograd zero_grad tests ---\n";
    nnf::Tensor zg = nnf::Tensor::from_vector({3}, {1});
    zg.set_requires_grad(true);
    (zg * 2.0f).backward();
    (zg * 2.0f).backward();
    std::cout << "after 2 backwards: " << zg.grad().to_string() << "\n";  // [4], blame piled up
    zg.zero_grad();
    std::cout << "after zero_grad: " << zg.grad().to_string() << "\n";    // [0], ready for the next step

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

