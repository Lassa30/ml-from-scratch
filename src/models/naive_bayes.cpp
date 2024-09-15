#include <models/naive_bayes.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace mlfs {

class GaussianNaiveBayes::Impl {
public:
  Impl() : isFitted_(false), labelsCnt_(0) {}

  void train(const Matrix &features, const Matrix &target) {
    // Exception handling
    if (features.shape().first != target.shape().first) {
      std::stringstream err_msg;
      err_msg << "Wrong shape of target or features:\n";
      err_msg << "target shape: (" << target.shape().first << ", " << target.shape().second << ")\n";
      err_msg << "features shape: (" << features.shape().first << ", " << features.shape().second << ")\n";
      throw std::logic_error(err_msg.str());
    }

    classLabels_ = get_labels(target);
    distsForFeatures_.resize(features.shape().second);
    labelsProbas_.resize(classLabels_.size());

    for (auto i = 0; i < features.shape().second; i++) {
      Matrix column = features.getCol(i);

      for (auto label : classLabels_) {
        if (i == 0) {
          labelsProbas_[label] += 1;
        }

        std::vector<double> choosen;
        for (auto j = 0; j < column.shape().first; ++j) {
          if (label == static_cast<int>(target.get(j, 0))) {
            choosen.push_back(column.get(j, 0));
          }
        }

        double mean = std::accumulate(choosen.begin(), choosen.end(), 0.0) / choosen.size();
        double stddev = 0.0;
        for (auto X : choosen) {
          stddev += (X - mean) * (X - mean);
        }

        if (choosen.size() == 1) {
          stddev = std::sqrt(choosen[0] / 1000);
        } else {
          stddev /= choosen.size() - 1;
          stddev = std::sqrt(stddev) + 4 * EPSILON;
        }

        distsForFeatures_[i].push_back(GaussianPDF(mean, stddev));
      }
    }

    for (auto &labProb : labelsProbas_) {
      labProb /= target.shape().first;
      labProb = std::log(labProb);
    }

    isFitted_ = true;
    labelsCnt_ = classLabels_.size();
  }

  Matrix predict(const Matrix &features) {
    if (!isFitted_) {
      throw std::logic_error("Couldn't predict, the model isn't fitted\n");
    }

    std::vector<double> prediction(features.shape().first);
    std::vector<double> probas(labelsProbas_);

    for (auto i = 0; i < features.rows(); ++i) {
      for (auto j = 0; j < features.cols(); ++j) {
        auto dist = distsForFeatures_[j];
        auto feature_i_j = features.get(i, j);

        for (auto label : classLabels_) {
          probas[label] += dist[label](feature_i_j);
        }
      }

      prediction[i] = std::max_element(probas.begin(), probas.end()) - probas.begin();
      probas = labelsProbas_;
    }

    return Matrix(prediction, Matrix::AXIS::ROW);
  }

private:
  class GaussianPDF {
  public:
    GaussianPDF(double mean, double stddev) : mean_(mean), std_(stddev) {}

    double operator()(double x) const {
      return -0.5 * std::log(2 * M_PI * std_ * std_) - (std::pow(x - mean_, 2) / (2 * std_ * std_));
    }

    double prob(double x) { return std::exp(operator()(x)); }

  private:
    double mean_;
    double std_;
  };

  std::set<int> get_labels(const Matrix &target) {
    std::set<int> unique_labels;
    for (auto i = 0; i < target.size(); i++) {
      unique_labels.insert(static_cast<int>(target.get(1, i)));
    }
    return unique_labels;
  }

  bool isFitted_;
  int labelsCnt_;
  std::set<int> classLabels_;
  std::vector<double> labelsProbas_;
  std::vector<std::vector<GaussianPDF>> distsForFeatures_;
};

GaussianNaiveBayes::GaussianNaiveBayes() : pImpl_(std::make_unique<Impl>()) {}
GaussianNaiveBayes::~GaussianNaiveBayes() = default;

void GaussianNaiveBayes::train(const Matrix &features, const Matrix &target) { pImpl_->train(features, target); }

Matrix GaussianNaiveBayes::predict(const Matrix &features) { return pImpl_->predict(features); }

} // namespace mlfs
