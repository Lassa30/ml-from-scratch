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

    virtual const MatrixXd& forward(const MatrixXd& y, const MatrixXd& y_pred) = 0;
    virtual const MatrixXd& backward() = 0;

  protected:
    MatrixXd y;   // input
    MatrixXd dy;  // gradient w.r.t. input
};

}  // namespace nn
}  // namespace mlfs