#include <models/naive_bayes.hpp>
#include <utils/matrix.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <random>
#include <set>

int main() {
  // DataSet is modified using python script. see: ../data
  std::vector<double> dataset;
  mlfs::utils::dataFromCsv(dataset, "../examples/data/IrisModified.csv");

  auto [features, target] = mlfs::utils::toDataset(dataset, 4);

  mlfs::Matrix designMatrixTrain(150, 4, features);
  mlfs::Matrix targetColumn(150, 1, target);

  mlfs::GaussianNaiveBayes naiveBayes{};
  naiveBayes.train(designMatrixTrain, targetColumn);

  auto prediction = naiveBayes.predict(designMatrixTrain);

  std::cout << "Actual prediction: " << '\n';
  prediction.printMatrix();
  std::cout << std::string(64, '-') << std::endl;

  std::cout << "Accuracy on train: " << mlfs::utils::accuracyScore(prediction, targetColumn) << '\n';

  std::cout << "\n\nTrain data prediction isn't the best way to check the model "
               "performance.\n"
            << "Let's split the data into train and test and evaluate the model.\n";

  // 150 * 0.7 = 105 -> 35 35 35
  // 150 * 0.3 = 45 -> 15 15 15

  std::set<int> datasetIdx;
  for (int i = 0; i < 150; i++) {
    datasetIdx.insert(i);
  }

  std::set<int> trainIdx;
  mlfs::utils::genIdx(trainIdx, 0, 49, 35);
  mlfs::utils::genIdx(trainIdx, 50, 99, 35);
  mlfs::utils::genIdx(trainIdx, 100, 149, 35);

  std::set<int> testIdx;

  std::set_difference(datasetIdx.begin(), datasetIdx.end(), trainIdx.begin(), trainIdx.end(),
                      std::inserter(testIdx, testIdx.begin()));

  std::vector<double> featuresTrain;
  mlfs::utils::vectFromIdx(featuresTrain, trainIdx, designMatrixTrain);
  std::vector<double> featuresTest;
  mlfs::utils::vectFromIdx(featuresTest, testIdx, designMatrixTrain);

  std::vector<double> targetTrain;
  mlfs::utils::vectFromIdx(targetTrain, trainIdx, targetColumn);
  std::vector<double> targetTest;
  mlfs::utils::vectFromIdx(targetTest, testIdx, targetColumn);

  mlfs::GaussianNaiveBayes NB{};

  designMatrixTrain = std::move(mlfs::Matrix(105, 4, featuresTrain));
  auto targetTrainColumn = std::move(mlfs::Matrix(105, 1, targetTrain));

  auto designMatrixTest = std::move(mlfs::Matrix(45, 4, featuresTest));
  auto targetTestColumn = std::move(mlfs::Matrix(45, 1, targetTest));

  NB.train(designMatrixTrain, targetTrainColumn);
  prediction = NB.predict(designMatrixTest);

  std::cout << "Prediction: " << '\n';
  prediction.printMatrix();
  std::cout << std::string(64, '-') << std::endl;

  std::cout << "Accuracy on test: " << mlfs::utils::accuracyScore(prediction, targetTestColumn) << "\n";
}