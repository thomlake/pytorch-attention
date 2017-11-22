from torch import FloatTensor
from torch.autograd import Variable
from torch.nn.functional import sigmoid, softmax


def mask3d(value, sizes):
    """Mask entries in value with 0 based on sizes.

    Args
    ----
    value: Tensor of size (B, N, D)
        Tensor to be masked. 
    sizes: list of int
        List giving the number of valid values for each item
        in the batch. Positions beyond each size will be masked.

    Returns
    -------
    value:
        Masked value.
    """
    v_mask = 0
    v_unmask = 1
    mask = value.data.new(value.size()).fill_(v_unmask)
    n = mask.size(1)
    for i, size in enumerate(sizes):
        if size < n:
            mask[i,size:,:] = v_mask
    return Variable(mask) * value


def fill_context_mask(mask, sizes, v_mask, v_unmask):
    """Fill attention mask inplace for a variable length context.

    Args
    ----
    mask: Tensor of size (B, N, D)
        Tensor to fill with mask values. 
    sizes: list[int]
        List giving the size of the context for each item in
        the batch. Positions beyond each size will be masked.
    v_mask: float
        Value to use for masked positions.
    v_unmask: float
        Value to use for unmasked positions.

    Returns
    -------
    mask:
        Filled with values in {v_mask, v_unmask}
    """
    mask.fill_(v_unmask)
    n_context = mask.size(2)
    for i, size in enumerate(sizes):
        if size < n_context:
            mask[i,:,size:] = v_mask
    return mask


def dot(a, b):
    """Compute the dot product between pairs of vectors in 3D Variables.
    
    Args
    ----
    a: Variable of size (B, M, D)
    b: Variable of size (B, N, D)
    
    Returns
    -------
    c: Variable of size (B, M, N)
        c[i,j,k] = dot(a[i,j], b[i,k])
    """
    return a.bmm(b.transpose(1, 2))


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
        combinations of the value vectors.
        Otherwise, the context vectors will be used.
    score: str or callable, default='dot'
        If score == 'dot', scores are computed
        as the dot product between context and
        query vectors. This Requires D1 == D2.
        Otherwise, score should be a callable:
             query    context     score
            (B,M,D1) (B,N,D2) -> (B,M,N)
    normalize: str, default='softmax'
        One of 'softmax', 'sigmoid', or 'identity'.
        Name of function used to map scores to weights.
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
        
    
    About
    -----
    Attention is used to focus processing on a particular region of input.
    This function implements the most common attention mechanism [1, 2, 3],
    which produces an output by taking a weighted combination of value vectors
    with weights from by a scoring function operating over pairs of query and
    context vectors.

    Given query vector `q`, context vectors `c_1,...,c_n`, and value vectors
    `v_1,...,v_n` the attention score of `q` with `c_i` is given by

        s_i = f(q, c_i)

    Frequently, `f` is given by the dot product between query and context vectors.

        s_i = q^T c_i

    The scores are passed through a normalization functions g.
    This is normally the softmax function.

        w_i = g(s_1,...,s_n)_i

    Finally, the output is computed as a weighted
    combination of the values with the normalized scores.

        z = sum_{i=1}^n w_i * v_i

    In many applications [4, 5] the context and value vectors are the same, `v_i = c_i`.

    Sizes
    -----
    This function accepts batches of size `B` containing
    `M` query vectors of dimension `D1`,
    `N` context vectors of dimension `D2`, 
    and optionally `N` value vectors of dimension `P`.

    Variable Length Contexts
    ------------------------    
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
    [1](https://arxiv.org/abs/1410.5401)
        @article{graves2014neural,
          title={Neural turing machines},
          author={Graves, Alex and Wayne, Greg and Danihelka, Ivo},
          journal={arXiv preprint arXiv:1410.5401},
          year={2014}
        }

    [2](https://arxiv.org/abs/1503.08895)

        @inproceedings{sukhbaatar2015end,
            title={End-to-end memory networks},
            author={Sukhbaatar, Sainbayar and Weston, Jason and Fergus, Rob and others},
            booktitle={Advances in neural information processing systems},
            pages={2440--2448},
            year={2015}
        }

    [3](https://distill.pub/2016/augmented-rnns/)

        @article{olah2016attention,
            title={Attention and augmented recurrent neural networks},
            author={Olah, Chris and Carter, Shan},
            journal={Distill},
            volume={1},
            number={9},
            pages={e1},
            year={2016}
        }

    [4](https://arxiv.org/abs/1409.0473)

        @article{bahdanau2014neural,
            title={Neural machine translation by jointly learning to align and translate},
            author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
            journal={arXiv preprint arXiv:1409.0473},
            year={2014}
        }

    [5](https://arxiv.org/abs/1506.03134)

        @inproceedings{vinyals2015pointer,
            title={Pointer networks},
            author={Vinyals, Oriol and Fortunato, Meire and Jaitly, Navdeep},
            booktitle={Advances in Neural Information Processing Systems},
            pages={2692--2700},
            year={2015}
        }
    """
    q, c, v = query, context, value
    if v is None:
        v = c

    batch_size_q, n_q, dim_q = q.size()
    batch_size_c, n_c, dim_c = c.size()
    batch_size_v, n_v, dim_v = v.size()

    if not (batch_size_q == batch_size_c == batch_size_v):
        msg = 'batch size mismatch (query: {}, context: {}, value: {})'
        raise ValueError(msg.format(q.size(), c.size(), v.size()))

    batch_size = batch_size_q

    # Compute scores
    if score == 'dot':
        s = dot(q, c)
    elif callable(score):
        s = score(q, c)
    else:
        raise ValueError('unknown score function "{}"'.format(f))

    # Normalize scores and mask contexts
    if normalize == 'softmax':
        if context_mask is not None:
            s = Variable(context_mask) + s
        elif context_sizes is not None:
            mask = s.data.new(batch_size, n_q, n_c)
            mask = fill_context_mask(mask, sizes=context_sizes, v_mask=float('-inf'), v_unmask=0)
            s = Variable(mask) + s

        s_flat = s.view(batch_size * n_q, n_c)
        w_flat = softmax(s_flat)
        w = w_flat.view(batch_size, n_q, n_c)

    elif normalize == 'sigmoid' or w == 'identity':
        w = sigmoid(s) if w == 'sigmoid' else s
        if context_mask is not None:
            w = Variable(context_mask) * w
        elif context_sizes is not None:
            mask = s.data.new(batch_size, n_q, n_c)
            mask = fill_context_mask(mask, sizes=context_sizes, v_mask=0, v_unmask=1)
            w = Variable(mask) * w

    else:
        raise ValueError('uknown normalize function "{}"'.format(normalize))

    # Combine
    z = w.bmm(v)
    if return_weight:
        return w, z
    return z
