#ifndef LAYERS_30_11_2024_HPP
#define LAYERS_30_11_2024_HPP

#include <eigen3/Eigen/Dense>

class Layer {
public:
  virtual ~Layer() = default;
  virtual MatrixXd &backward(const MatrixXd &input) = 0;
  virtual MatrixXd &getOutput(const MatrixXd &input) = 0;
};

class ActivationLayer : public Layer {};

class FullyConnectedLayer : public Layer {
public:
  // Optimizer uses these methods to update weights and biases
  virtual std::vector<MatrixXd> &getWeights() = 0;
  virtual std::vector<double> &getBiases() = 0;

  virtual std::vector<MatrixXd> &getWeightsGrads(const MatrixXd &input) = 0;
  virtual std::vector<double> &getBiasesGrads(const MatrixXd &input) = 0;
};

class Linear : public FullyConnectedLayer;

class SoftMax : public ActivationLayer;
class RELU : public ActivationLayer;

#endif