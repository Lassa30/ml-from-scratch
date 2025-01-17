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

class Model {
 public:
  Model();

  // TODO: come up with a convinient way to construct models
  // Example:
  // auto m = Model(Layer(3, 5, "l1"), ReLU("l2"), Layer(5, 2, "l3"), Sigmoid());
  // m.addResiduals({"l1", "l3"});

  virtual ~Model();
  virtual const MatrixXd& forward(const MatrixXd& x);

  // TODO: save and load methods for a model.
  // void save();
  // void load();
 private:
  class ModelStructure {
   private:
    class LayerNode;

   public:
    void addLayer();
  };

  ModelStructure model_;
};

}  // namespace nn
}  // namespace mlfs

#endif  // MODEL_HPP_2025_01_11