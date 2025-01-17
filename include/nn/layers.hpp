#ifndef LAYERS_30_11_2024_HPP
#define LAYERS_30_11_2024_HPP

#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <string>

namespace mlfs {
namespace nn {

using namespace Eigen;

class Layer {
 public:
  virtual ~Layer() = default;

  virtual const MatrixXd& forward(const MatrixXd& X) = 0;
  virtual const MatrixXd& backward(const MatrixXd& X) = 0;

  virtual const std::string& getId() const noexcept final;

 protected:
  std::string id_;
};

class ActivationLayer : public Layer {};

class ParametricLayer : public Layer {
 public:
  virtual const MatrixXd& getWeights() = 0;
  virtual const MatrixXd& getWeightsGrad(const MatrixXd& X) = 0;

  virtual double getBiases() = 0;
  virtual double getBiasesGrad(const MatrixXd& X) = 0;
};

class Linear : public ParametricLayer {
 public:
  Linear(const std::int64_t& in, const std::int64_t& out, std::string id);

  const MatrixXd& getWeights() override;
  const double& getBiases() override;

  const MatrixXd& getWeightsGrad(const MatrixXd& X) override;
  const double& getBiasesGrad(const MatrixXd& X) override;

 private:
  std::int64_t inputShape_;
  std::int64_t outputShape_;
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