#include <models/naive_bayes.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>

namespace mlfs {

// Gaussian PDF in point x: return -0.5 * (2 * M_PI * std_ * std_).log() - (x - mean_).square() / (2 * std_ * std_));
class GaussianNaiveBayes::Impl {
  class GaussianPDF {
  public:
    GaussianPDF() = default;
    GaussianPDF(const MatrixXd &mean, const MatrixXd &stddev, const int &labelsCnt)
        : mean_(mean), stdSquared_(stddev), labelsCnt_(labelsCnt) {}

    GaussianPDF &operator=(const GaussianPDF &rhs) {
      mean_ = rhs.mean_;
      stdSquared_ = rhs.stdSquared_;
      labelsCnt_ = rhs.labelsCnt_;
      return *this;
    }

    MatrixXd operator()(const MatrixXd &x) const {
      if (!(x.cols() == mean_.cols() && x.cols() == stdSquared_.cols())) {
        throw std::invalid_argument("Wrong data for GaussianPDF: features number don't match.\n");
      }

      MatrixXd logScore(x.rows(), labelsCnt_);
      for (int ind = 0; ind < labelsCnt_; ind++) {
        auto stdRowSq = stdSquared_.row(ind).array() + 4 * __DBL_EPSILON__;
        auto preLogScore =
            (((x.array().rowwise() - mean_.row(ind).array()).square()).rowwise() / (-2.0 * stdRowSq)).rowwise() -
            0.5 * ((2.0 * M_PI * stdRowSq).log());

        logScore.col(ind) = preLogScore.rowwise().sum();
      }
      return logScore;
    }

  private:
    MatrixXd mean_;
    MatrixXd stdSquared_;
    int labelsCnt_;
  };

  std::set<int> get_labels(const MatrixXd &target) {
    std::set<int> unique_labels;
    for (auto i = 0; i < target.size(); i++) {
      unique_labels.insert(static_cast<int>(target(i, 0)));
    }
    return unique_labels;
  }

  VectorXd countLabels(const MatrixXd &target) {
    // assume that target is a column-vector
    if (target.cols() != 1) {
      throw std::invalid_argument("GaussianNaiveBayes::Impl\n\tcountLabels: wrong target shape.\n");
    }
    std::map<int, int> labelsCounterMap;
    int maxLabel = -1;
    for (int labelIndex = 0; labelIndex < target.rows(); labelIndex++) {
      labelsCounterMap[int(target(labelIndex))] += 1;
    }

    labelsCnt_ = labelsCounterMap.size();
    VectorXd labelsCounter(labelsCnt_);
    for (auto [label, labelCount] : labelsCounterMap) {
      labelsCounter[label] = labelCount;
    }
    return labelsCounter;
  }

  bool isFitted_;
  int labelsCnt_;
  std::set<int> classLabels_;
  VectorXd labelsLogProbas_;
  GaussianPDF gaussianPDF_;

public:
  Impl() : isFitted_{false}, labelsCnt_{}, labelsLogProbas_{}, gaussianPDF_{} {};

  void train(const MatrixXd &features, const MatrixXd &target) {
    classLabels_ = get_labels(target);
    labelsCnt_ = classLabels_.size();
    // labelsProbas_ = [how many labels of each class] / [number of rows]
    labelsLogProbas_ = (countLabels(target) / target.rows()).array().log();

    std::vector<int> indexesOfLabel;
    MatrixXd featureMeans(labelsCnt_, features.cols());
    MatrixXd featureStd(labelsCnt_, features.cols());

    for (auto label : classLabels_) {
      // TODO: move to separate function
      for (auto j = 0; j < features.rows(); ++j) {
        if (label == static_cast<int>(target(j, 0))) {
          indexesOfLabel.push_back(j);
        }
      }

      MatrixXd selectedRows = features(indexesOfLabel, placeholders::all);

      featureMeans.row(label) = selectedRows.colwise().mean();
      auto divide_by = (indexesOfLabel.size() - 1 + 4 * __DBL_EPSILON__);
      featureStd.row(label) =
          ((selectedRows.rowwise() - featureMeans.row(label)).array().square().colwise().sum() / divide_by);
      indexesOfLabel.clear();
    }

    gaussianPDF_ = GaussianPDF(featureMeans, featureStd, labelsCnt_);
    isFitted_ = true;
  }

  MatrixXd predict(const MatrixXd &features) {
    if (!isFitted_) {
      throw std::logic_error("\nCouldn't predict, the model isn't fitted\n");
    }

    MatrixXd prediction(features.rows(), 1);
    MatrixXd log_scores(features.rows(), labelsCnt_);

    log_scores = predict_proba(features);
    for (int row = 0; row < features.rows(); row++) {
      log_scores.row(row).maxCoeff(&prediction(row, 0));
    }
    return prediction;
  }

  MatrixXd predict_proba(const MatrixXd &features) const {
    if (!isFitted_) {
      throw std::logic_error("\nCouldn't predict, the model isn't fitted\n");
    }
    MatrixXd logProba(features.rows(), labelsCnt_);
    VectorXd Px(features.rows());
    logProba = gaussianPDF_(features).rowwise() + labelsLogProbas_.transpose();

    for (int i = 0; i < logProba.rows(); ++i) {
        double maxLog = logProba.row(i).maxCoeff();
        logProba.row(i) = (logProba.row(i).array() - maxLog).exp();
        Px(i) = logProba.row(i).sum();
        logProba.row(i) /= Px(i);
    }
    return logProba;
    return logProba;
  }
};

GaussianNaiveBayes::GaussianNaiveBayes() : pImpl_(std::make_unique<Impl>()) {}

GaussianNaiveBayes::~GaussianNaiveBayes() = default;

void GaussianNaiveBayes::train(const MatrixXd &features, const MatrixXd &target) { pImpl_->train(features, target); }

MatrixXd GaussianNaiveBayes::predict(const MatrixXd &features) const { return pImpl_->predict(features); }

MatrixXd GaussianNaiveBayes::predict_proba(const MatrixXd &features) const { return pImpl_->predict_proba(features); }

} // namespace mlfs
