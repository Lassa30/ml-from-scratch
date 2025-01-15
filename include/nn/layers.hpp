#ifndef LAYERS_30_11_2024_HPP
#define LAYERS_30_11_2024_HPP

#include <eigen3/Eigen/Dense>

namespace mlfs {
namespace nn {

using namespace Eigen;

class Layer {
 public:
  virtual ~Layer() = default;
  virtual MatrixXd &backward(const MatrixXd &input) = 0;
  virtual MatrixXd &forward(const MatrixXd &input) = 0;
};

class ActivationLayer : public Layer {};

class Linear : public Layer {
 public:
  Linear();
  // Optimizer uses these methods to update weights and biases
  const MatrixXd &getWeights();
  const double &getBiases();

  MatrixXd &getWeightsGrad(const MatrixXd &input);
  double &getBiasesGrad(const MatrixXd &input);

 private:
  MatrixXd X;  // "X" is a good name for an input, right?
  MatrixXd W;  // Weights
  double b;    // bias term

  // Layer' output gradients
  MatrixXd dX;  // w.r.t. input
  MatrixXd dW;  // w.r.t. weights
  double db;    // w.r.t. bias
};

// TODO: Dense Layer with activation function as a parameter

// class SoftMax : public ActivationLayer {
// };

// class RELU : public ActivationLayer {
// };

// class Sigmoid : public ActivationLayer {
// };

}  // namespace nn
}  // namespace mlfs

#endif