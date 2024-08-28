#ifndef NAIVE_BAYES_HPP_08_28_24
#define NAIVE_BAYES_HPP_08_28_24


#include "../utils/matrix.hpp"
#include <random>
#include <set>

namespace mlfs {

using namespace mlfs;

// Naive Bayes Classifier that uses the Gaussian PDF as its core.
class GaussianNaiveBayes {
public:

    template <typename T>
    void train(const Matrix & features, const Matrix & target) {

        // Exception handling
        if (features.shape().first != target.shape().first) {
            std::stringstream err_msg;

            err_msg << "Wrong shape of target or features:\n";
            err_msg << "target shape: (" << target.shape().first << ", " << target.shape().second << ")\n";
            err_msg << "features shape: (" << features.shape().first << ", " << features.shape().second << ")\n";

            throw std::logic_error(
                err_msg.str()
            );
        }

        // Core

        std::vector<T> classLabels = get_labels(target);
        for (auto i : get_labels) {

            for (auto i = 0; i < features.shape().second; i++) {
                Matrix column = features.get_col(i);

                // HERE!!!
            }
        }
        

        isFitted_ = true;
    }

    Matrix predict(const Matrix & features) {
        // nothing to predict
        if (!isFitted_) {
            throw std::logic_error(
                "Couldn't predict, the model isn't fitted\n"
            );
        }



    }   


private:

    bool isFitted_ = 0;

    // Not computationally stable, log-likelihood is better.
    std::vector<std::normal_distribution<double>> dists_ = {}; 

    template<typename T>
    vector<T> get_labels(const Matrix & target) {
        std::set<T> unique_labels;

        for (auto i = 0; i < target.size(); i++) {
            unique_labels.insert(target[i]);
        }

        return std::vector<T>(unique_labels.begin(), unique_labels.end());
    }

};

// Naive Bayes Classifier for Multinominally distributed data | used in text classification
class MultinominalNaiveBayes {

};

} // namespace mlfs

#endif // NAIVE_BAYES_HPP_08_28_24