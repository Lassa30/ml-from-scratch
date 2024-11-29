#ifndef NAIVE_BAYES_HPP_08_28_24
#define NAIVE_BAYES_HPP_08_28_24

#include <eigen3/Eigen/Dense>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mlfs {

using namespace Eigen;

class GaussianNaiveBayes {
public:
  GaussianNaiveBayes();
  ~GaussianNaiveBayes();

  void train(const MatrixXd &features, const MatrixXd &target);
  MatrixXd predict(const MatrixXd &features) const;
  MatrixXd predict_proba(const MatrixXd &features) const;

private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
};

} // namespace mlfs

#endif // NAIVE_BAYES_HPP_08_28_24
