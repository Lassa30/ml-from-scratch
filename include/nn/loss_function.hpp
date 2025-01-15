#include <algorithm>
#include <layers.hpp>
#include <memory>
#include <model.cpp>
#include <random>
#include <set>
#include <vector>

namespace mlfs {
namespace nn {

class LossFunction {
 public:
  virtual ~LossFunction() = default;

  virtual const MatrixXd& operator()(const MatrixXd& y, const MatrixXd& y_pred) = 0;
  virtual const MatrixXd& backward() = 0;

  virtual MatrixXd operator+=() final;

 protected:
  MatrixXd X;   // input
  MatrixXd dX;  // gradient w.r.t. input
};

}  // namespace nn
}  // namespace mlfs