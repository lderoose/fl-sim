# Terminology

- **classical**: non-federated training mode. A single model is trained on the full dataset using standard mini-batch gradient descent, without any master-worker architecture or gradient compression.

- **epoch**: a full training cycle during which all workers collectively process their entire training data once. An epoch consists of multiple *steps*.

- **step**: a single gradient update. At each step, each participating worker processes one mini-batch of its local data, sends its gradient to the master, which aggregates them and broadcasts the updated model weights back.

- **dense**: the original tensor in its original dimension

e.g.
```python
t = tensor(
    [
        [0, 1],
        [4, 9],
        [1, 3],
        [3, 5]
    ]
)
```

- **compressed**: the compressed format represents the 'dense' tensor in a lower dimension. The number of rows of the compressed tensor depend to the parameters of the compressor, but the number of columns is always 2:
    - the first column is the index of the value in a 'flat' representation 
    - the second column is the value.

e.g.
```python
c = tensor(
    [
        [3., 9.],
        [7., 5.]
    ]
)
```

- **sparse**: the sparse format represents the 'compressed' tensor in the original dimension (dimension of the dense tensor). All missing values are filled with 0.

e.g.
```python
s = tensor(
    [
        [0., 0.],
        [0., 9.],
        [0., 0.],
        [0., 5.]
    ]
)
```
