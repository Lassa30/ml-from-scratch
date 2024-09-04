#ifndef NAIVE_BAYES_HPP_08_28_24
#define NAIVE_BAYES_HPP_08_28_24

#include <utils/matrix.hpp>

#include <random>
#include <set>
#include <map>

namespace mlfs
{

    class GaussianPDF
    {
    private:
        double mean_ = 0;
        double std_ = 0;

    public:
        GaussianPDF(double mean, double stddev)
            : mean_{mean}, std_{stddev} {};

        double operator()(double x) const
        {
            return -0.5 * std::log(2 * M_PI * std_ * std_) - (std::pow(x - mean_, 2) / (2 * std_ * std_));
        }

        double prob(double x)
        {
            return std::exp(operator()(x));
        }
    };

    // Naive Bayes Classifier that uses the Gaussian PDF as its core.
    class GaussianNaiveBayes
    {
    public:
        GaussianNaiveBayes() = default;
        ~GaussianNaiveBayes() = default;

        void train(const Matrix &features, const Matrix &target)
        {

            // Exception handling
            if (features.shape().first != target.shape().first)
            {
                std::stringstream err_msg;

                err_msg << "Wrong shape of target or features:\n";
                err_msg << "target shape: (" << target.shape().first << ", " << target.shape().second << ")\n";
                err_msg << "features shape: (" << features.shape().first << ", " << features.shape().second << ")\n";

                throw std::logic_error(
                    err_msg.str());
            }

            // Go through each feature-column and
            //      if target == Class_label_i then add it to the new vector
            //      then count std and mean for this vector
            //      vuala: you can build a Gaussian PDF

            classLabels_ = get_labels(target);
            distsForFeatures_.resize(features.shape().second);
            labelsProbas_.resize(classLabels_.size());

            for (auto i = 0; i < features.shape().second; i++)
            {
                Matrix column = features.get_col(i);

                for (auto label : classLabels_)
                {
                    // counting labels
                    if (i == 0)
                    {
                        labelsProbas_[label] += 1;
                    }

                    // choosing suitable data from the feature column
                    std::vector<double> choosen;

                    for (auto j = 0; j < column.shape().first; ++j)
                    {
                        if (label == static_cast<int>(target.get(j, 0)))
                        {
                            choosen.push_back(column.get(j, 0));
                        }
                    }

                    double mean =
                        std::accumulate(choosen.begin(), choosen.end(), 0.0) / choosen.size();

                    double stddev = 0.0;
                    for (auto X : choosen)
                        stddev += (X - mean) * (X - mean);

                    // If we have only one choosen element everything will go the unpreffered way...

                    // adding small constant for numerical stability is important
                    if (choosen.size() == 1)
                        stddev = std::sqrt(choosen[0] / 1000);
                    else
                    {
                        stddev /= choosen.size() - 1;
                        stddev = std::sqrt(stddev) + 4 * EPSILON;
                    }

                    distsForFeatures_[i].push_back(
                        GaussianPDF(mean, stddev));
                }
            }

            // Getting probabilities for each label:
            // LabelProba =   ( LabelCount )
            //              -------------------
            //               ( NumberOfLabels )

            for (auto &labProb : labelsProbas_)
            {
                labProb /= target.shape().first;
                labProb = std::log(labProb);
            }

            // The model is fitted now.
            isFitted_ = true;
            labelsCnt_ = classLabels_.size();
        }

        Matrix predict(const Matrix &features)
        {
            // nothing to predict
            if (!isFitted_)
            {
                throw std::logic_error(
                    "Couldn't predict, the model isn't fitted\n");
            }

            // 1) go through feature matrix and for each feature in the row
            //    calculate density(from pre-built Gaussian PDFs)
            // 2) for each feature save calculations as the map: <class_label: proba>
            // 3) find argmax and add it to the prediction vector
            //    return Matrix(features.shape().first, 1, prediction)

            vector<double> prediction(features.shape().first);

            std::vector<double> probas(labelsProbas_);

            for (auto i = 0; i < features.rows_; ++i)
            {

                for (auto j = 0; j < features.cols_; ++j)
                {
                    auto dist = distsForFeatures_[j];

                    auto feature_i_j = features.get(i, j);

                    for (auto label : classLabels_)
                    {
                        probas[label] += dist[label](feature_i_j);
                    }
                }

                prediction[i] = std::max_element(probas.begin(), probas.end()) - probas.begin();
                probas = labelsProbas_;
            }

            return Matrix(prediction, Matrix::AXIS::ROW);
        }

    private:
        bool isFitted_ = 0;

        int labelsCnt_ = 0;
        std::set<int> classLabels_;
        std::vector<double> labelsProbas_;

        // C * F - total amount of distributions
        // C - class labels count
        // F - features count

        vector<vector<GaussianPDF>> distsForFeatures_ = {};

        std::set<int> get_labels(const Matrix &target)
        {
            std::set<int> unique_labels;

            for (auto i = 0; i < target.size(); i++)
            {
                unique_labels.insert(static_cast<int>(target.get(1, i)));
            }

            return unique_labels;
        }
    };

    // Naive Bayes Classifier for Multinominally distributed data | used in text classification
    class MultinominalNaiveBayes
    {
    public:
    };

} // namespace mlfs

#endif // NAIVE_BAYES_HPP_08_28_24