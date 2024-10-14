#include <models/neighbors.hpp>

#include <algorithm>
#include <cmath>

namespace mlfs {

class KNN::Impl {
public:
 Impl() = default;

 Impl(int k_neighbors) {
   if (k_neighbors <= 0) {
     throw std::invalid_argument(
         "KNN(int k_neighbors):\n\t"
         "The amount of neighbors should be greater than 0.n");
   }
   k_neighbors_ = k_neighbors;
  }

  void train(const Matrix &features, const Matrix &target) {
    if (k_neighbors_ > target.rows()) {
      k_neighbors_ = target.rows();
    }
    features_ = features;
    target_ = target;

    auto targetVector = target_.getData();
    classes_ = std::unique(targetVector.begin(), targetVector.end()) - targetVector.begin();
  }

  Matrix predict_proba(const Matrix &features, MetricsKNN metric = euclidean) {
    // allocate memory for predictions - once
    std::vector<std::pair<double, int>> distances(features_.rows());
    std::vector<int> class_labels(k_neighbors_);
    std::vector<double> class_labels_probas(classes_);
    std::vector<double> results(features.rows() * classes_);

    for (int i = 0; i < features.rows(); i++) {
      auto currentRow = features.getRow(i);

      // finding distances and saving
      for (int j = 0; j < features_.rows(); j++) {
        distances[j] = {find_distance(currentRow, features_.getRow(j), metric), target_.get(j, 1)};
      }

      // sorting by distances
      std::sort(
          distances.begin(), distances.end(),
          [](const std::pair<double, int> &lhs, const std::pair<double, int> &rhs) { return lhs.first < rhs.first; });

      // assign labels from sorted array
      for (int ind = 0; ind < k_neighbors_; ind++) {
        class_labels[ind] = distances[ind].second;
      }

      for (int ind = 0; ind < classes_; ind++) {
        class_labels_probas[ind] = std::count(class_labels.begin(), class_labels.end(), ind) / k_neighbors_;
      }

      std::copy(class_labels_probas.begin(), class_labels_probas.end(),
                results.begin() + i * classes_);
    }
    return Matrix(features.rows(), classes_, results);
  }

  Matrix predict(const Matrix &features, MetricsKNN metric = euclidean) {
    auto predictedProbas = predict_proba(features, metric);
    std::vector<double> prediction(features.rows());
    for (int ind = 0; ind < features.rows(); ind++) {
      auto probaVect = predictedProbas.getRow(ind).getData();
      auto argmax = std::distance(probaVect.begin(), std::max_element(probaVect.begin(), probaVect.end()));
      prediction[ind] = argmax;
    }
    return Matrix(features.rows(), 1, prediction);
  }

private:
  int k_neighbors_{5};

  Matrix features_;
  Matrix target_;
  int classes_;

  double find_distance(const Matrix &lhs, const Matrix &rhs, MetricsKNN metric) {
    double distance = 0.0;
    if (metric == euclidean) {
      distance = (lhs * lhs - rhs * rhs).sum();
    } else if (metric == cosine) {
      distance = 1 - (lhs * rhs).sum() / std::sqrt(lhs.L2()) / std::sqrt(rhs.L2());
    } else if (metric == manhattan) {
      distance = abs(lhs - rhs).sum();
    } else {
      throw std::invalid_argument("KNN\n\tThe provided metric isn't the member of \"MetricsKNN\" "
                                  "class.");
    }
    return distance;
  }
};

KNN::KNN() : pImpl_(std::make_unique<Impl>()){};
KNN::KNN(int k_neighbors) : pImpl_(std::make_unique<Impl>(k_neighbors)) {}
KNN::~KNN() = default;

void KNN::train(const Matrix &features, const Matrix &target) { pImpl_->train(features, target); }

Matrix KNN::predict_proba(const Matrix &features, MetricsKNN metric = euclidean) {
  return pImpl_->predict_proba(features, metric);
}

Matrix KNN::predict(const Matrix &features, MetricsKNN metric = euclidean) {
  return pImpl_->predict(features, metric);
}
} // namespace mlfs