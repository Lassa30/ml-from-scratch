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
    virtual const MatrixXd& backward(const MatrixXd& prevGrad) = 0;

    virtual const std::string& getId() const noexcept final { return id_; }

  protected:
    std::string id_;
};

class ActivationLayer : public Layer {};

class ParametricLayer : public Layer {
  public:
    virtual const MatrixXd& getWeights() = 0;
    virtual const MatrixXd& computeWeightsGrad(const MatrixXd& prevGrad) = 0;

    virtual double getBiases() = 0;
    virtual double computeBiasesGrad(const MatrixXd& prevGrad) = 0;
};

class Linear : public ParametricLayer {
  public:
    Linear(const std::int64_t& in, const std::int64_t& out, const std::string& id);

    const MatrixXd& getWeights() override;
    double getBiases() override;

    const MatrixXd& computeWeightsGrad(const MatrixXd& prevGrad) override;
    double computeBiasesGrad(const MatrixXd& prevGrad) override;

  private:
    std::int64_t inputShape_;
    std::int64_t outputShape_;

    MatrixXd X_;
    MatrixXd W_;
    double b_;

    MatrixXd dX_;
    MatrixXd dW_;
    double db_;
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