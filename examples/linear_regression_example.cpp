#include <models/linear_regression.hpp>
#include <utils/utils.hpp>

#include <iomanip>
#include <memory>

int main() {
  std::cout << std::fixed;

  std::vector<double> data;
  mlfs::utils::dataFromCsv(data, "../examples/data/winequality-white.csv", ';');

  auto [X, y] = std::move(mlfs::utils::toDataset(data, 11));

  mlfs::Matrix designMatrix(4898, 11, X);
  mlfs::Matrix targetColumn(4898, 1, y);

  // White wine dataset: Linear Regression using SGD and MSE as a loss function
  auto opt = std::make_unique<mlfs::optim::SGD>(3.5e-5);
  auto loss = std::make_unique<mlfs::optim::MSE>();

  std::cout << "lr: " << opt->getLearningRate() << '\n';

  mlfs::LinearRegression LinReg(std::move(opt), std::move(loss));

  LinReg.train(designMatrix, targetColumn, 1024, 1000);

  auto prediction = LinReg.predict(designMatrix);

  std::cout << std::setprecision(6);
  std::cout << "Compare some predictions to the target:\n";
  for (int elem = 0; elem < 10; ++elem)
    std::cout << prediction.get(elem, 1) << " | " << targetColumn.get(elem, 1) << '\n';

  std::cout << "SCORE:\n\t";
  std::cout << LinReg.score(prediction, targetColumn) << std::endl;

  // Using Lasso LinReg
  std::cout << "LassoLinReg:\n";
  opt = std::make_unique<mlfs::optim::SGD>(2e-5);
  loss = std::make_unique<mlfs::optim::MSE>(mlfs::optim::Reg::L1, 1e-3);
  std::cout << "lr: " << opt->getLearningRate() << '\n';

  mlfs::LinearRegression LassoReg(std::move(opt), std::move(loss));

  LassoReg.train(designMatrix, targetColumn, 1024, 1000);

  prediction = std::move(LassoReg.predict(designMatrix));

  std::cout << "Compare some predictions to the target:\n";
  for (int elem = 0; elem < 10; ++elem)
    std::cout << prediction.get(elem, 1) << " | " << targetColumn.get(elem, 1) << '\n';

  std::cout << "SCORE:\n\t";
  std::cout << LassoReg.score(prediction, targetColumn) << std::endl;

  std::cout << "\nLASSO REGRESSION weights:\n";
  LassoReg.printWeights();

  // Using Ridge Linear Regression
  std::cout << "RidgeLinReg:\n";
  opt = std::make_unique<mlfs::optim::SGD>(2.5e-5);
  loss = std::make_unique<mlfs::optim::MSE>(mlfs::optim::Reg::L2, 1e-3);
  std::cout << "lr: " << opt->getLearningRate() << '\n';

  mlfs::LinearRegression RidgeReg(std::move(opt), std::move(loss));

  RidgeReg.train(designMatrix, targetColumn, 1024, 1000);

  prediction = std::move(RidgeReg.predict(designMatrix));

  std::cout << "Compare some predictions to the target:\n";
  for (int elem = 0; elem < 10; ++elem)
    std::cout << prediction.get(elem, 1) << " | " << targetColumn.get(elem, 1) << '\n';

  std::cout << "SCORE:\n\t";
  std::cout << RidgeReg.score(prediction, targetColumn) << std::endl;

  std::cout << "\nRIDGE REGRESSION weights:\n";
  RidgeReg.printWeights();
}
