#include <utils/matrix.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace mlfs {
// Linear Regression with no regularization and MSE as a loss function.
class LinearRegressionSGD {
public:
  LinearRegressionSGD() = default;
  ~LinearRegressionSGD() = default;

  LinearRegressionSGD(const std::size_t &batch, const double &learningRate,
                      const int &epochs)
      : learning_rate_{learningRate}, batch_{batch}, epochs_{epochs} {}

  void train(Matrix &features, Matrix &target, int randomState = 42) {
    // random
    std::mt19937 gen(randomState);
    std::uniform_int_distribution<> intDis(0, features.rows() - 1);
    std::uniform_real_distribution<> realDis(-1, 1);

    // preparation
    batch_ = std::min(batch_, features.rows());
    weights_ =
        Matrix(1, features.cols(), std::vector<double>(features.cols(), 0));
    bias_ = 0.0;

    // SGD
    for (int e = 0; e < epochs_; ++e) {

      for (auto i = 0; i < features.rows(); i += batch_) {
        std::vector<int> idx(batch_);
        std::generate(idx.begin(), idx.end(), [&]() { return intDis(gen); });

        Matrix featuresBatch = getBatch(features, idx);
        Matrix targetBatch = getBatch(target, idx);

        Matrix dw =
            gradMSE(targetBatch, featuresBatch); // Gradient w.r.t. weights
        double db =
            biasGradMSE(targetBatch, featuresBatch); // Gradient w.r.t. bias

        weights_ -= dw * learning_rate_; // weights update
        bias_ -= db * learning_rate_;    // bias update
      }
      
    }
  }

  Matrix predict(const Matrix &features) {
    return features.matmul(weights_.T()) + bias_;
  }

  double score(const Matrix &prediction, const Matrix &target) {
    return ((target - prediction) * (target - prediction) / target.rows())
        .sum();
  }

  void printWeights() const {
    std::cout << "\nWEIGHTS:\n";
    weights_.print_matrix();

    std::cout << "\nBIAS:\n";
    std::cout << bias_ << std::endl;
  }

private:
  double learning_rate_ = 0.001;
  std::size_t batch_ = 10;
  int epochs_ = 10;

  Matrix weights_{};
  double bias_{};

  int randomState_ = 42;

  // MSE differentiation w.r.t. weights_
  Matrix gradMSE(const Matrix &y, const Matrix &X) const {
    // dw = -(2 / len(y_batch)) * np.dot(X_batch.T, (y_batch - y_pred))
    // db = -(2 / len(y_batch)) * np.sum(y_batch - y_pred)
    auto y_pred = X.matmul(weights_.T()) + bias_;
    auto grad = (X.T().matmul(y - y_pred)) * -2.0 / batch_;

    return grad.T();
  }

  double biasGradMSE(const Matrix &y, const Matrix &X) const {
    auto y_pred = X.matmul(weights_.T()) + bias_;

    return ((y - y_pred) * (-2.0) / batch_).sum();
  }

  Matrix getBatch(Matrix &mat, const std::vector<int> &idx) const {
    std::vector<double> resVect;
    for (auto rowInd : idx) {
      auto rowVect = mat.get_row(rowInd).get_data();
      resVect.insert(resVect.end(), rowVect.begin(), rowVect.end());
    }

    if (resVect.size() == batch_ * mat.cols()) {
      return Matrix(batch_, mat.cols(), resVect);
    } else {
      throw std::runtime_error("getBatch():\n\tVect size don't match...\n");
    }
  }
};
} // namespace mlfs