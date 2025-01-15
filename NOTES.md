# Just notes for me

---

## Training

I want to use this way of training a model.

```c++
auto optimizer = SGD(/*lr=*/0.001, model->parameters())
// ...
auto loss = MSE(); // Derived class of LossFunction
lossFn += lossFn(target, model->forward());
// ... 
optimizer.zero_grad();

optimizer.backward(lossFn);

optimizer.step(lossFn);
```

### Optimizer and LossFunction connection
- optimizer.zero_grad() -> make all grads of Layer' stored in LayerGrad equal to zero
- optimizer.backward() -> calculates grads firstly for a loss function and then for layers
- optimizer.step() -> update weights using calculated gradients


<!-- ## Where to store weights?
There are two ways I see:
1) Inside of the layer. Layer has backward public method where it calculates it's grads and gives them "back".
2) Inside of the Layer Node 
As far as I can see the right answer is in the Layer, it will be more logical.
-->