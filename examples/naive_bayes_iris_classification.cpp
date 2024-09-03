#include <models/naive_bayes.hpp>
#include <utils/utils.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <set>
#include <algorithm>
#include <random>

#include <set>
#include <random>

void genIdx(std::set<int>& idxSet, const int left, const int right, const int cnt, int randomState = 42) {
    int setSize = 0;

    std::mt19937 gen(randomState);
    std::uniform_int_distribution<> dis(left, right);

    while (setSize < cnt) {
        int num = dis(gen);
        if (idxSet.insert(num).second) {
            ++setSize;
        }
    }
}

void vectFromIdx(std::vector<double>& vect, const std::set<int>& setIdx, const mlfs::Matrix& designMatrixTrain) {
    for (auto i : setIdx) {
        std::vector<double> toPush(designMatrixTrain.get_row(i).get_data());
        vect.insert(vect.end(), toPush.begin(), toPush.end());
    }
}

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

    std::cout << "Accuracy on train: " << mlfs::utils::accuracyScore(prediction, targetColumn) << '\n';

    std::cout << "\n\nTrain data prediction isn't the best way to check the model performance.\n" \
              << "Let's split the data into train and test and evaluate the model.\n";

    // 150 * 0.7 = 105 -> 35 35 35 
    // 150 * 0.3 = 45 -> 15 15 15 
    
    std::set<int> datasetIdx;
    for (int i = 0; i < 150; i++) {
        datasetIdx.insert(i);
    }

    std::set<int> trainIdx;
    genIdx(trainIdx, 0, 49, 35);
    genIdx(trainIdx, 50, 99, 35);
    genIdx(trainIdx, 100, 149, 35);

    std::set<int> testIdx;

    // std::cout << "1) The train set has size: " << trainIdx.size() << '\n';
    // for (auto i : trainIdx) std::cout << i << ' ';
    // std::cout << '\n';

    std::set_difference(datasetIdx.begin(), datasetIdx.end(),
                        trainIdx.begin(), trainIdx.end(),
                        std::inserter(testIdx, testIdx.begin())
    );

    // std::cout << "2) The test set has size: " << testIdx.size() << '\n';
    // for (auto i : testIdx) std::cout << i << ' ';
    // std::cout << '\n';

    std::vector<double> featuresTrain;
    vectFromIdx(featuresTrain, trainIdx, designMatrixTrain);
    std::vector<double> featuresTest;
    vectFromIdx(featuresTest, testIdx, designMatrixTrain);

    std::vector<double> targetTrain;
    vectFromIdx(targetTrain, trainIdx, targetColumn);
    std::vector<double> targetTest;
    vectFromIdx(targetTest, testIdx, targetColumn);

    mlfs::GaussianNaiveBayes NB{};
    
    designMatrixTrain = std::move(mlfs::Matrix(105, 4, featuresTrain));
    auto targetTrainColumn = std::move(mlfs::Matrix(105, 1, targetTrain));

    auto designMatrixTest = std::move(mlfs::Matrix(45, 4, featuresTest));
    auto targetTestColumn = std::move(mlfs::Matrix(45, 1, targetTest));

    NB.train(designMatrixTrain, targetTrainColumn);
    prediction = NB.predict(designMatrixTest);

    std::cout << "Prediction: " << '\n';
    prediction.print_matrix();
    std::cout << std::string(64, '-') << std::endl;

    std::cout << "Accuracy on test: " 
              << mlfs::utils::accuracyScore(prediction, targetTestColumn) 
              << "\n\n";

}