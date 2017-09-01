```python
def attend(
    query, context, value=None, f='dot', 
    query_sizes=None, query_mask=None,
    context_sizes=None, context_mask=None,
    return_weight=False):
    """Attend to value (or context) by scoring each query and context.

    Args
    ----
    query: Variable of size (B, M, D1)
        Batch of M query vectors.
    context: Variable of size (B, N, D2)
        Batch of N context vectors.
    value: Variable of size (B, N, P), default=None
        If given, the output vectors will be convex
        combinations of the value vectors. Otherwise,
        the context vectors will be used.
    f: callable or str, default='dot'
        If f == 'dot' use dot product attention.
        In this case D1 must be equal to D2.
        Otherwise, f should be a callable that given query
        and context returns a Variable of shape (B, M, N).
    query_mask: Tensor of (B, M, N), default=None
        A Tensor to use to mask query. Unmasked entries
        should be 0 and masked entries should be -inf.
    query_sizes: list[int], default=None,
        List giving the size of query for each item in the
        batch. If query_mask or query_sizes are not given,
        context is assumed to have fixed size.
    context_mask: Tensor of (B, M, N), default=None
        A Tensor to use to mask context. Unmasked entries
        should be 1 and masked entries should be 0.
    context_sizes: list[int], default=None,
        List giving the size of context for each item in the
        batch. If context_mask or context_sizes are not given,
        context is assumed to have fixed size.
    return_weight: bool, default=False
        If True, return the attention weight Tensor.

    Returns
    -------
    output: Variable of size (B, M, P)
        If return_weight is False.
    weight, output: Variable of size (B, M, N), Variable of size (B, M, P)
        If return_weight is True.
    """
```

About
-----
Attention is used to focus processing on a particular region of input.
The `attend` function provided by this package implements the most
common attention mechanism [[1](#1), [2](#2), [3](#3)], which produces
an output by taking a convex combination of value vectors with weights
from by a scoring function operating over pairs of query and context vectors.

Given query vector `q`, context vectors `c_1,...,c_n`, and value vectors
`v_1,...,v_n` the attention score of `q` with `c_i` is given by

    s_i = f(q, c_i)

Frequently `f` is given by the dot product between query and context vectors.

    s_i = q^T c_i

The scores are normalized using a softmax function.

    w_i = softmax(s_1,...,s_n)_i

Finally, the output is computed as a convex combination
of the values with the normalized score weights.

    z = sum_{i=1}^n w_i * v_i

In many applications [[4](#4), [5](#5)] attention is applied
to the context vectors themselves, `v_i = c_i`.

Sizes
-----
This `attend` function provided by this package accepts
batches of size `B` containing
`M` query vectors of dimension `D1`, 
`N` context vectors of dimension `D2`, 
and optionally `N` value vectors of dimension `P`.

Variable Length
---------------
If the number of context vectors varies within a batch,
a context can be ignored by adding negative infinity to
the corresponding score. This will cause the softmax to
evaluate to zero at those locations. Likewise, query vectors
can be ignored by multiplying their normalized score by zero.

A context mask, with entries in `{-inf, 0}`, and query mask,
with entries in `{0, 1}`, can be passed into the `attend` function.
The masks should have size `(B, M, N)`.

Alternatively, lists can be passed giving the size of the query
and context for each item in the batch. Appropriate masks will
be created from these lists.

References
----------
#### [[1]](https://arxiv.org/abs/1410.5401)
    @article{graves2014neural,
      title={Neural turing machines},
      author={Graves, Alex and Wayne, Greg and Danihelka, Ivo},
      journal={arXiv preprint arXiv:1410.5401},
      year={2014}
    }

#### [[2]](https://arxiv.org/abs/1503.08895)

    @inproceedings{sukhbaatar2015end,
        title={End-to-end memory networks},
        author={Sukhbaatar, Sainbayar and Weston, Jason and Fergus, Rob and others},
        booktitle={Advances in neural information processing systems},
        pages={2440--2448},
        year={2015}
    }

#### [[3]](https://distill.pub/2016/augmented-rnns/)

    @article{olah2016attention,
        title={Attention and augmented recurrent neural networks},
        author={Olah, Chris and Carter, Shan},
        journal={Distill},
        volume={1},
        number={9},
        pages={e1},
        year={2016}
    }

#### [[4]](https://arxiv.org/abs/1409.0473)

    @article{bahdanau2014neural,
        title={Neural machine translation by jointly learning to align and translate},
        author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1409.0473},
        year={2014}
    }

#### [[5]](https://arxiv.org/abs/1506.03134)

    @inproceedings{vinyals2015pointer,
        title={Pointer networks},
        author={Vinyals, Oriol and Fortunato, Meire and Jaitly, Navdeep},
        booktitle={Advances in Neural Information Processing Systems},
        pages={2692--2700},
        year={2015}
    }
