#ifndef KNN_HPP_2024_10_14_17_26
#define KNN_HPP_2024_10_14_17_26

#include <memory>
#include <utils/matrix.hpp>

namespace mlfs {

enum MetricsKNN { euclidean, cosine, manhattan };

class KNN {
public:
  KNN();
  KNN(int k_neighbors);
  ~KNN();

  void train(const Matrix &features, const Matrix &target);
  Matrix predict_proba(const Matrix &features, MetricsKNN metric);
  Matrix predict(const Matrix &features, MetricsKNN metric);

private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
};

} // namespace mlfs

#endif