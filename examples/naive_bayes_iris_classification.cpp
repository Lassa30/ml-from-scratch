#include <models/naive_bayes.hpp>
#include <utils/utils.hpp>

#include <vector>
#include <string>
#include <iostream>


int main() {
    // DataSet is modified using python script. see: ../data
    std::vector<double> dataset;
    mlfs::utils::dataFromCsv(dataset, "../examples/data/IrisModified.csv");
    std::vector<double> features;
    std::vector<double> target;

    for (int i = 0; i < dataset.size(); ++i) {
        if (i % 5 == 4) {
            target.push_back(dataset[i]);
        }else {
            features.push_back(dataset[i]);
        }
    }

    mlfs::Matrix designMatrixTrain(150, 4, features);
    mlfs::Matrix targetColumn(150, 1, target);

    mlfs::GaussianNaiveBayes naiveBayes{};
    naiveBayes.train(designMatrixTrain, targetColumn);

    auto prediction = naiveBayes.predict(designMatrixTrain);

    std::cout << "Actual prediction: " << '\n';
    prediction.print_matrix();
    std::cout << std::string(64, '-') << std::endl;

    std::cout << "Accuracy on train set: " << mlfs::utils::accuracyScore(prediction, targetColumn) << '\n';
}