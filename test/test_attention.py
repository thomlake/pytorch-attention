import numpy as np
import pytest

import torch
from torch import FloatTensor
from torch.autograd import Variable

from attention import attention


def Volatile(x):
    return Variable(x, volatile=True)


def test_fill_query_mask():
    batch_size, n_q, n_c = 3, 4, 5
    query_sizes = [4, 3, 2]
    context_sizes = [3, 2, 5]
    mask = attention.fill_query_mask(FloatTensor(batch_size, n_q, n_c), sizes=query_sizes)

    for i in range(batch_size):
        for j in range(n_q):
            for k in range(n_c):
                if j < query_sizes[i]:
                    assert mask[i,j,k] == 1
                else:
                    assert mask[i,j,k] == 0

def test_fill_context_mask():
    batch_size, n_q, n_c = 3, 4, 5
    query_sizes = [4, 3, 2]
    context_sizes = [3, 2, 5]
    mask = attention.fill_context_mask(FloatTensor(batch_size, n_q, n_c), sizes=context_sizes)

    for i in range(batch_size):
        for j in range(n_q):
            for k in range(n_c):
                if k < context_sizes[i]:
                    assert mask[i,j,k] == 0
                else:
                    assert mask[i,j,k] == -float('inf')


def test_dot():
    batch_size, n_q, n_c, d = 31, 18, 15, 22
    q = np.random.normal(0, 1, (batch_size, n_q, d))
    c = np.random.normal(0, 1, (batch_size, n_c, d))
    s = attention.dot(
        Volatile(torch.from_numpy(q)),
        Volatile(torch.from_numpy(c))).data.numpy()

    assert s.shape == (batch_size, n_q, n_c)
    for i in range(batch_size):
        for j in range(n_q):
            for k in range(n_c):
                assert np.allclose(np.dot(q[i,j], c[i,k]), s[i,j,k])


@pytest.mark.parametrize(
    'batch_size,n_q,n_c,d', [
    (1, 1, 6, 11),
    (20, 1, 10, 3),
    (3, 10, 15, 5)])
def test_attention(batch_size, n_q, n_c, d):
    q = np.random.normal(0, 1, (batch_size, n_q, d))
    c = np.random.normal(0, 1, (batch_size, n_c, d))
    w_out, z_out = attention.attend(
            Volatile(torch.from_numpy(q)),
            Volatile(torch.from_numpy(c)), return_weight=True)
    w_out = w_out.data.numpy()
    z_out = z_out.data.numpy()

    assert w_out.shape == (batch_size, n_q, n_c)
    assert z_out.shape == (batch_size, n_q, d)

    for i in range(batch_size):
        for j in range(n_q):
            s = [np.dot(q[i,j], c[i,k]) for k in range(n_c)]
            max_s = max(s)
            exp_s = [np.exp(si - max_s) for si in s]
            sum_exp_s = sum(exp_s)

            w_ref = [ei / sum_exp_s for ei in exp_s]
            assert np.allclose(w_ref, w_out[i,j])

            z_ref = sum(w_ref[k] * c[i,k] for k in range(n_c))
            assert np.allclose(z_ref, z_out[i,j])


@pytest.mark.parametrize(
    'batch_size,n_q,n_c,d,p', [
    (1, 1, 6, 11, 5),
    (20, 1, 10, 3, 14),
    (3, 10, 15, 5, 9)])
def test_attention_values(batch_size, n_q, n_c, d, p):
    q = np.random.normal(0, 1, (batch_size, n_q, d))
    c = np.random.normal(0, 1, (batch_size, n_c, d))
    v = np.random.normal(0, 1, (batch_size, n_c, p))
    w_out, z_out = attention.attend(
            Volatile(torch.from_numpy(q)),
            Volatile(torch.from_numpy(c)),
            value=Volatile(torch.from_numpy(v)), return_weight=True)
    w_out = w_out.data.numpy()
    z_out = z_out.data.numpy()

    assert w_out.shape == (batch_size, n_q, n_c)
    assert z_out.shape == (batch_size, n_q, p)

    for i in range(batch_size):
        for j in range(n_q):
            s = [np.dot(q[i,j], c[i,k]) for k in range(n_c)]
            max_s = max(s)
            exp_s = [np.exp(si - max_s) for si in s]
            sum_exp_s = sum(exp_s)

            w_ref = [ei / sum_exp_s for ei in exp_s]
            assert np.allclose(w_ref, w_out[i,j])

            z_ref = sum(w_ref[k] * v[i,k] for k in range(n_c))
            assert np.allclose(z_ref, z_out[i,j])


@pytest.mark.parametrize(
    'batch_size,n_q,n_c,d,query_sizes,context_sizes', [
    (1, 1, 6, 11, None, [3]),
    (4, 1, 10, 3, None,[7, 5, 10, 9]),
    (3, 10, 15, 5, [10, 7, 2], None),
    (2, 5, 11, 7, [3, 5], [9, 2])])
def test_attention_masked(batch_size, n_q, n_c, d, query_sizes, context_sizes):
    q = np.random.normal(0, 1, (batch_size, n_q, d))
    c = np.random.normal(0, 1, (batch_size, n_c, d))

    w_out, z_out = attention.attend(
        Volatile(torch.from_numpy(q)),
        Volatile(torch.from_numpy(c)),
        query_sizes=query_sizes, context_sizes=context_sizes, return_weight=True)
    w_out = w_out.data.numpy()
    z_out = z_out.data.numpy()

    assert w_out.shape == (batch_size, n_q, n_c)
    assert z_out.shape == (batch_size, n_q, d)

    w_checked = np.zeros((batch_size, n_q, n_c), dtype=int)
    z_checked = np.zeros((batch_size, n_q, d), dtype=int)

    for i in range(batch_size):
        for j in range(n_q):
            n_qi = query_sizes[i] if query_sizes is not None else n_q
            n_ci = context_sizes[i] if context_sizes is not None else n_c

            s = [np.dot(q[i,j], c[i,k]) for k in range(n_ci)]
            max_s = max(s)
            exp_s = [np.exp(sk - max_s) for sk in s]
            sum_exp_s = sum(exp_s)

            w_ref = [ek / sum_exp_s for ek in exp_s]
            for k in range(n_c):
                if j < n_qi and k < n_ci:
                    assert np.allclose(w_ref[k], w_out[i,j,k])
                    w_checked[i,j,k] = 1
                else:
                    assert np.allclose(0, w_out[i,j,k])
                    w_checked[i,j,k] = 1

            z_ref = sum(w_ref[k] * c[i,k] for k in range(n_ci))
            if j < n_qi:
                for k in range(d):
                    assert np.allclose(z_ref[k], z_out[i,j,k])
                    z_checked[i,j,k] = 1
            else:
                for k in range(d):
                    assert np.allclose(0, z_out[i,j,k])   
                    z_checked[i,j,k] = 1

    assert np.all(w_checked == 1)
    assert np.all(z_checked == 1)
