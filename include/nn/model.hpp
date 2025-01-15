#ifndef MODEL_HPP_2025_01_11
#define MODEL_HPP_2025_01_11

#include <eigen3/Eigen/Dense>
#include <map>
#include <memory>
#include <nn/layers.hpp>
#include <string>

namespace mlfs {
namespace nn {

using namespace Eigen;
using LayerPtr = std::shared_ptr<Layer>;

// DONE: new name for the project is needed - it's not ML from scratch now (I use Eigen).
// It's SimplyML now...

/// TODO: think about creating a base model class where user' "forward" implementation is needed

class Model {
 public:
  Model();

  virtual ~Model();

  virtual const MatrixXd& forward(const MatrixXd& x);

  // TODO: save and load methods for a model.
  // void save();
  // void load();

  // class to store connections of each Layer
  // I consider it is a possible "autograd" alternative
  class LayerGraph {
   private:
    class LayerNode;
  };

  virtual LayerGraph& parameters();
};

// class Model {
// }

}  // namespace nn
}  // namespace mlfs

#endif  // MODEL_HPP_2025_01_11