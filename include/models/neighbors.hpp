#ifndef KNN_HPP_2024_10_14_17_26
#define KNN_HPP_2024_10_14_17_26

#include <eigen3/Eigen/Dense>
#include <memory>

namespace mlfs {

using namespace Eigen;

enum MetricsKNN { euclidean, cosine, manhattan };

class KNN {
  public:
    KNN();
    KNN(int k_neighbors);
    ~KNN();

    void train(const MatrixXd& features, const MatrixXd& target);
    MatrixXd predict_proba(const MatrixXd& features, MetricsKNN metric);
    MatrixXd predict(const MatrixXd& features, MetricsKNN metric);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

}  // namespace mlfs

#endif