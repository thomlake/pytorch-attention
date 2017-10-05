```python
def attend(
        query, context, value=None,
        score='dot', normalize='softmax',
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
        If given, the output vectors will be weighted
        combinations of the value vectors. Otherwise,
        the context vectors will be used.
    score: str or callable, default='dot'
        If score == 'dot', scores are computed
        as the dot product between context and
        query vectors. This Requires D1 == D2.
        Otherwise, score should be a callable:
               query  context      score
            (B,M,D1) (B,N,D2) -> (B,M,N)
    normalize: str, default='softmax'
        One of 'softmax', 'sigmoid', or 'identity'.
        Function used to map scores to weights.
    context_mask: Tensor of (B, M, N), default=None
        A Tensor used to mask context. Masked
        and unmasked entries should be filled 
        appropriately for the normalization function.
    context_sizes: list[int], default=None,
        List giving the size of context for each item
        in the batch and used to compute a context_mask.
        If context_mask or context_sizes are not given,
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
common attention mechanism [[1](#1), [2](#2), [3](#3), [4](#4)], which produces
an output by taking a weighted combination of value vectors with weights
from a scoring function operating over pairs of query and context vectors.

Given query vector <img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/e73485aa867794d51ccd8725055d03a3.svg?invert_in_darkmode" align=middle width=9.93993pt height=14.55729pt/>, context vectors <img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/39d2a848a943a7f5ec27272dad27c784.svg?invert_in_darkmode" align=middle width=68.70633pt height=14.55729pt/>, and value vectors
<img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/1eb39a281b1e66935a51005b6beb9dbe.svg?invert_in_darkmode" align=middle width=72.878355pt height=14.55729pt/> the attention score of <img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/e73485aa867794d51ccd8725055d03a3.svg?invert_in_darkmode" align=middle width=9.93993pt height=14.55729pt/> with <img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/28e6b84adb66aca59d04ec9e227bfd3f.svg?invert_in_darkmode" align=middle width=13.00398pt height=14.55729pt/> is given by

<p align="center"><img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/a5a09669219f681bb51e176b190b0e4a.svg?invert_in_darkmode" align=middle width=88.61622pt height=16.376943pt/></p>

Frequently <img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> is given by the dot product between query and context vectors.

<p align="center"><img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/39c9d05724010ea29be9eb321b1422ec.svg?invert_in_darkmode" align=middle width=68.318745pt height=17.805315pt/></p>

The scores are passed through a normalization functions <img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.398995pt height=14.10255pt/> (normally the softmax function).

<p align="center"><img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/a5d4c0a87edcc90e9dc7bb8a1845e86a.svg?invert_in_darkmode" align=middle width=124.74033pt height=16.376943pt/></p>

Finally, the output is computed as a weighted sum
of the value vectors.

<p align="center"><img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/5397f1268e113895a997a61e51165ffc.svg?invert_in_darkmode" align=middle width=103.98927pt height=44.878845pt/></p>

In many applications [[1](#1), [4](#4), [5](#5)] attention is applied
to the context vectors themselves, <img src="https://rawgit.com/thomlake/pytorch-attention/None/svgs/da2cf8b162672dc46adcace06ec2740a.svg?invert_in_darkmode" align=middle width=50.28639pt height=14.55729pt/>.

Sizes
-----
This `attend` function provided by this package accepts
batches of size `B` containing
`M` query vectors of dimension `D1`, 
`N` context vectors of dimension `D2`, 
and optionally `N` value vectors of dimension `P`.

Variable Length
---------------
If the number of context vectors varies within a batch, a context
can be ignored by forcing the corresponding weight to be zero.

In the case of the softmax, this can be achieved by adding negative
infinity to the corresponding score before normalization.
Similarly, for elementwise normalization functions the weights can
be multiplied by an appropriate {0,1} mask after normalization.

To facilitate the above behavior, a context mask, with entries
in `{-inf, 0}` or `{0, 1}` depending on the normalization function,
can be passed to this function. The masks should have size `(B, M, N)`.

Alternatively, a list can be passed giving the size of the context for
each item in the batch. Appropriate masks will be created from these lists.

Note that the size of output does not depend on the number of context vectors.
Because of this context positions are truly unaccounted for in the output.

References
----------
#### [[1]](https://arxiv.org/abs/1409.0473)

    @article{bahdanau2014neural,
        title={Neural machine translation by jointly learning to align and translate},
        author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1409.0473},
        year={2014}
    }

#### [[2]](https://arxiv.org/abs/1410.5401)
    @article{graves2014neural,
      title={Neural turing machines},
      author={Graves, Alex and Wayne, Greg and Danihelka, Ivo},
      journal={arXiv preprint arXiv:1410.5401},
      year={2014}
    }

#### [[3]](https://arxiv.org/abs/1503.08895)

    @inproceedings{sukhbaatar2015end,
        title={End-to-end memory networks},
        author={Sukhbaatar, Sainbayar and Weston, Jason and Fergus, Rob and others},
        booktitle={Advances in neural information processing systems},
        pages={2440--2448},
        year={2015}
    }

#### [[4]](https://distill.pub/2016/augmented-rnns/)

    @article{olah2016attention,
        title={Attention and augmented recurrent neural networks},
        author={Olah, Chris and Carter, Shan},
        journal={Distill},
        volume={1},
        number={9},
        pages={e1},
        year={2016}
    }

#### [[5]](https://arxiv.org/abs/1506.03134)

    @inproceedings{vinyals2015pointer,
        title={Pointer networks},
        author={Vinyals, Oriol and Fortunato, Meire and Jaitly, Navdeep},
        booktitle={Advances in Neural Information Processing Systems},
        pages={2692--2700},
        year={2015}
    }
