// #ifndef MODEL_HPP_2025_01_11
// #define MODEL_HPP_2025_01_11

// #include <map>
// #include <memory>
// #include <nn/layers.hpp>
// #include <string>

// namespace mlfs {
// namespace nn {

// using namespace Eigen;
// using LayerPtr = std::unique_ptr<Layer>;

// class Model {
//   public:
//     Model();

//     // TODO: come up with a convinient way to construct models
//     // Example:
//     // auto m = Model(Layer(3, 5, "l1"), ReLU("l2"), Layer(5, 2, "l3"), Sigmoid());
//     // m.addResiduals({"l1", "l3"});

//     virtual ~Model();
//     virtual const MatrixXd& forward(const MatrixXd& x);
//     void applyOptimizer(std::unique_ptr<Optimizer> optimzizer);

//   private:
//     class ModelStructure {
//       private:
//         class LayerNode {
//           public:
//             LayerPtr forward();
//             LayerPtr backward();

//           private:
//             LayerPtr next_;
//             LayerPtr prev_;
//             std::vector<LayerPtr> residuals_;

//             LayerPtr layer_;

//             void nextForResiduals();
//             void prevForResiduals();
//         };

//       public:
//         void applyOptimizer(std::unique_ptr<Optimizer> optimizer);
//         void addLayer(const Layer& layer);
//     };

//     ModelStructure model_;
// };

// }  // namespace nn
// }  // namespace mlfs

// #endif  // MODEL_HPP_2025_01_11