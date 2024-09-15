#ifndef NAIVE_BAYES_HPP_08_28_24
#define NAIVE_BAYES_HPP_08_28_24

#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utils/matrix.hpp>
#include <vector>

namespace mlfs {

class GaussianNaiveBayes {
public:
  GaussianNaiveBayes();
  ~GaussianNaiveBayes();

  void train(const Matrix &features, const Matrix &target);
  Matrix predict(const Matrix &features);

private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
};

} // namespace mlfs

#endif // NAIVE_BAYES_HPP_08_28_24
