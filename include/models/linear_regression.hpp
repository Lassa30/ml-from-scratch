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

namespace mlfs
{
    // Linear Regression with no regularization and MSE as a loss function.
    class LinearRegressionSGD
    {
    public:
        LinearRegressionSGD() = default;
        ~LinearRegressionSGD() = default;

        LinearRegressionSGD(const std::size_t &batch, const double &learningRate, const int &epochs)
            : lr_{learningRate}, batch_{batch}, epochs_{epochs}
        {
        }

        void train(const Matrix &features, const Matrix &target, int randomState = 42)
        {
            // random
            std::mt19937 gen(randomState);
            std::uniform_int_distribution<> intDis(0, features.rows_ - 1);
            std::uniform_real_distribution<> realDis(-1, 1);

            // preparation
            batch_ = std::min(batch_, features.rows_);
            // std::vector<double> randVect(features.cols_);
            // for (auto &elem : randVect)
            // {
            //     elem = realDis(gen);
            // }
            weights_ = Matrix(1, features.cols_, std::vector<double>(features.cols_, 0));
            bias_ = 0.0;

            // SGD

            for (int e = 0; e < epochs_; ++e)
            {
                for (auto i = 0; i <= features.rows_; i += batch_)
                {
                    // full rewrite needed
                }
            }
        }

        Matrix predict(const Matrix &features)
        {
            // nothing to predict
            // if (!isFitted_)
            // {
            //     throw std::logic_error(
            //         "Linear Regression: Couldn't predict, the model isn't fitted\n");
            // }
            auto mat = features.matmul(weights_.T());
            mat += bias_;

            return mat;
        }

        double score(const Matrix &prediction, const Matrix &target)
        {
            return ((target - prediction) * (target - prediction) / target.rows_).sum();
        }

        void printWeights() const
        {
            std::cout << "\nWEIGHTS\n";
            weights_.print_matrix();
            std::cout << "\nBIAS\n";
            std::cout << bias_ << std::endl;
        }

    private:
        double lr_ = 0.001;
        std::size_t batch_ = 10;
        int epochs_ = 10;
        bool isFitted_ = false;

        Matrix weights_{};
        double bias_{};

        int randomState_ = 42;

        // SGD
        Matrix MSE(const Matrix &y, const Matrix &X, const Matrix &w_0, const double &b) const
        {
            auto prediction = std::move(X.matmul(w_0.T()) + b);
            return (y - prediction) * (y - prediction) / X.rows_;
        }

        // MSE differentiation w.r.t. weights_
        Matrix gradMSE(const Matrix &y, const Matrix &X) const
        {
            // dw = -(2 / len(y_batch)) * np.dot(X_batch.T, (y_batch - y_pred))
            // db = -(2 / len(y_batch)) * np.sum(y_batch - y_pred)

            auto y_pred = X.matmul(weights_.T()) + bias_;
            auto XT = X.T();
            auto diff = y - y_pred;
            auto gradScaled = XT.matmul(diff) * (-2.0) / batch_;

            return gradScaled.T();
        }

        double biasGradMSE(const Matrix &y, const Matrix &X) const
        {
            auto y_pred = X.matmul(weights_.T()) + bias_;

            auto gradScaled = (y - y_pred) * (-2.0) / batch_;

            return gradScaled.sum();
        }

        Matrix getBatch(const Matrix &mat, const std::vector<int> &idx) const
        {
            std::vector<double> resVect;
            for (auto rowInd : idx)
            {
                auto rowVect = mat.get_row(rowInd).get_data();
                resVect.insert(resVect.end(), rowVect.begin(), rowVect.end());
            }

            if (resVect.size() == batch_ * mat.cols_)
            {
                return Matrix(batch_, mat.cols_, resVect);
            }

            else
            {
                throw std::runtime_error("getBatch():\n\tVect size don't match...\n");
            }
        }
    };
}