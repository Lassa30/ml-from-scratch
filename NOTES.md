## Just some random stuff
>Could curiously recurring template pattern `CRTP` be used in mlfs?

>Do you need to store a reference to an owner in `AnythingImpl` classes?
Answer: probably not, remove later if it is true

>Generating a documentation from code comments is essential
### Brief development plan
1) ~~*Implement*: basic shape and stride operations~~
2) *Implement*: `tensor.reshape(Shape)` through changing the `Shape` and `Stride` \
also remember to check if the reshaping is possible
3) *Implement*: `tensor.resize(Shape)` through changing the `data_` vector in the `Storage`
4) *Design and Implement*: `tensor.view(<something>)` idk how exactly.
5) *Design and Implement*: `TensorIterator` to iterate the tensor with different `Shape`s, `Stride`s and `offset`s
6) *Implement*: an add and multiply methods with the overloads and broadcasting.
7) *Design and Implement*: a dummy autograd with summation and multiplication operations only