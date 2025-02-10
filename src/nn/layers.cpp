// #include <nn/layers.hpp>

// // Linear::Linear(std::int64_t in, std::int64_t out, const std::string& id)
// //     : inputShape_{in}, outputShape_{out}, id_{id}
// // {
// //   W_ = MatrixXd::Zero(outputShape_, inputShape_);
// //   b_ = db_ = 0;
// // }

// // const MatrixXd& Linear::forward(const MatrixXd& X) {
// //   if (X.cols() != inputShape_) {
// //     throw std::invalid_argument("forward: Wrong shape\n");
// //   }
// //   X_ = X;
// //   return (W_ * X.transpose()).rowwise() + b_;
// // }

// // const MatrixXd& Linear::backward(const MatrixXd& prevGrad) {
// //   // ...
// // };

// // const MatrixXd& Linear::getWeights() { return W_; };
// // double Linear::getBiases() { return b_; }

// // const MatrixXd& Linear::computeWeightsGrad(const MatrixXd& prevGrad) { return prevGrad; }
// // const double Linear::computeBiasesGrad(const MatrixXd& prevGrad) { return
// // prevGrad.rowwise().sum(); }