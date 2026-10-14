import threading
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import pytest
import torch
from event_transformer import EventImageEncoder


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2.))
    def forward(self,pixel_values):
        hidden = pixel_values.flatten(2).transpose(1,2)[:,:5,:]*self.gain
        return SimpleNamespace(last_hidden_state=hidden)


def make_encoder(device):
    encoder = object.__new__(EventImageEncoder)
    encoder.device = torch.device(device)
    encoder.network = Network().to(device).eval()
    encoder._projection_owner = encoder.network
    encoder._trace_lock = threading.Lock()
    encoder._trace_checked = False
    encoder._single_projection = None
    return encoder


def pixels():
    return np.random.default_rng(967).random((2,3,224,224),dtype=np.float32)


def test_cpu_keeps_original_forward_without_tracing():
    encoder = make_encoder('cpu')
    data = pixels()
    with patch('event_transformer.torch.jit.trace',side_effect=AssertionError('traced CPU')):
        actual = encoder.encode_prepared([data])
    with torch.inference_mode():
        expected = encoder._eager_vectors(torch.from_numpy(data)).reshape(1,-1).numpy()
    np.testing.assert_array_equal(actual,expected)
    assert not encoder._trace_checked


mps = pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')


@mps
def test_fixed_single_trace_is_bit_exact_and_keeps_inplace_weight_updates():
    encoder = make_encoder('mps')
    data = pixels()
    actual = encoder.encode_prepared([data])
    assert encoder._single_projection is not None
    with torch.inference_mode():
        expected = encoder._eager_vectors(torch.from_numpy(data).to('mps')).reshape(1,-1).cpu().numpy()
        encoder.network.gain.fill_(3.)
        changed = encoder._eager_vectors(torch.from_numpy(data).to('mps')).reshape(1,-1).cpu().numpy()
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(encoder.encode_prepared([data]),changed)


@mps
def test_offline_batch_and_training_do_not_trace():
    encoder = make_encoder('mps')
    data = pixels()
    with patch('event_transformer.torch.jit.trace',side_effect=AssertionError('traced batch')):
        actual = encoder.encode_prepared([data,data])
        encoder.network.train()
        single = encoder.encode_prepared([data])
    assert actual.shape==(2,12) and single.shape==(1,12)
    assert not encoder._trace_checked


@mps
def test_trace_failure_returns_exact_original_result_and_does_not_retry():
    encoder = make_encoder('mps')
    data = pixels()
    with patch('event_transformer.torch.jit.trace',side_effect=RuntimeError('unsupported operator')) as trace:
        first = encoder.encode_prepared([data])
        second = encoder.encode_prepared([data])
    assert trace.call_count==1 and encoder._single_projection is None
    np.testing.assert_array_equal(first,second)


@mps
def test_trace_numerical_difference_keeps_eager_inference():
    encoder = make_encoder('mps')
    data = pixels()
    with patch('event_transformer.torch.jit.trace',return_value=lambda tensor:torch.zeros((2,6),device='mps')):
        actual = encoder.encode_prepared([data])
    assert encoder._trace_checked and encoder._single_projection is None
    with torch.inference_mode():
        expected = encoder._eager_vectors(torch.from_numpy(data).to('mps')).reshape(1,-1).cpu().numpy()
    np.testing.assert_array_equal(actual,expected)


@mps
def test_replacing_encoder_network_keeps_eager_inference():
    encoder = make_encoder('mps')
    data = pixels()
    encoder.encode_prepared([data])
    encoder.network = Network().to('mps').eval()
    with patch('event_transformer.torch.jit.trace',side_effect=AssertionError('traced new owner')):
        actual = encoder.encode_prepared([data])
    with torch.inference_mode():
        expected = encoder._eager_vectors(torch.from_numpy(data).to('mps')).reshape(1,-1).cpu().numpy()
    np.testing.assert_array_equal(actual,expected)


@mps
def test_runtime_trace_failure_disables_cached_graph_before_exact_fallback():
    encoder = make_encoder('mps')
    data = pixels()
    expected = encoder.encode_prepared([data])
    encoder._single_projection = Mock(side_effect=RuntimeError('graph failure'))
    actual = encoder.encode_prepared([data])
    assert encoder._single_projection is None
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(encoder.encode_prepared([data]),expected)
