"""Tests for wrapper helpers for making DataStreams play nicely with PyTorch DataLoaders.
"""
# Standard
from unittest import mock

# Third Party
from torch.utils.data._utils import worker

# First Party
from caikit.core.data_model import DataStream

# Local
from caikit_nlp.toolkit.data_stream_wrapper import SimpleIterableStreamWrapper
from tests.fixtures import requires_determinism

# Sample data to load via PyTorch
SAMPLE_DATA = [{"label": "foo"}, {"label": "foo"}, {"label": "bar"}, {"label": "bar"}]
SAMPLE_STREAM = DataStream.from_iterable(SAMPLE_DATA)
NUM_CYCLES = 10


def test_without_shuffling():
    """Ensure that we can build a datastream & load it in a data loader without shuffling."""
    test_results = []
    # Get the IDs of all objects in the stream
    get_stream_id_order = lambda s: [id(datum) for datum in s]
    # Compare the data stream at two different iteration points; here, we
    # produce True if two streams have the same objects in the same order
    have_same_id_order = lambda id_set1, id_set2: all(
        [datum1 == datum2 for datum1, datum2 in zip(id_set1, id_set2)]
    ) and len(id_set1) == len(id_set2)

    # NOTE - a buffer size of 1 is a noop; the shuffle operation just gets the current element
    wrapper = SimpleIterableStreamWrapper(stream=SAMPLE_STREAM, shuffle=False)
    # Cycle through NUM_CYCLES times & ensure that the order of our objects does not change
    initialize_order = get_stream_id_order(wrapper)
    for _ in range(NUM_CYCLES):
        cycle_ids = get_stream_id_order(wrapper)
        test_res = have_same_id_order(initialize_order, cycle_ids)
        test_results.append(test_res)
    assert all(test_results)


def test_shuffle_full_buffer(requires_determinism):
    """Ensure that we can build a datastream & shuffle it all in memory on each iteration."""
    test_results = []
    # Get the IDs of all objects in the stream
    get_stream_id_order = lambda s: [id(datum) for datum in s]
    # Compare the data stream at two different iteration points; here, we
    # produce True if two streams have the same objects in the same order
    have_same_id_order = lambda id_set1, id_set2: all(
        [datum1 == datum2 for datum1, datum2 in zip(id_set1, id_set2)]
    ) and len(id_set1) == len(id_set2)

    # NOTE - a buffer size of 1 is a noop; the shuffle operation just gets the current element
    wrapper = SimpleIterableStreamWrapper(
        stream=SAMPLE_STREAM, shuffle=True, buffer_size=len(SAMPLE_STREAM)
    )
    # Cycle through NUM_CYCLES times & ensure that the order of our objects DOES change sometimes
    initialize_order = get_stream_id_order(wrapper)
    for _ in range(NUM_CYCLES):
        cycle_ids = get_stream_id_order(wrapper)
        test_res = have_same_id_order(initialize_order, cycle_ids)
        test_results.append(test_res)
    assert not all(test_results)


def test_iter_with_multi_worker():
    """Ensure that we are able to iterate properly over data in case of workers
    managed by torch"""
    test_results = []
    w1_info = worker.WorkerInfo(id=0, num_workers=3, seed=7)
    w2_info = worker.WorkerInfo(id=1, num_workers=3, seed=7)
    w3_info = worker.WorkerInfo(id=2, num_workers=3, seed=7)
    # Worker distribution works round robin after we consider shuffling.
    # Since we don't shuffle in this patched test, they should just be
    # divided as is.
    index_stream = [
        {"label": 0},  # goes to worker 0
        {"label": 1},  # goes to worker 1
        {"label": 2},  # goes to worker 2
        {"label": 3},  # goes to worker 0
        {"label": 4},  # goes to worker 1
        {"label": 5},  # goes to worker 2
    ]
    worker_info = [
        (w1_info, [index_stream[0], index_stream[3]]),
        (w2_info, [index_stream[1], index_stream[4]]),
        (w3_info, [index_stream[2], index_stream[5]]),
    ]
    for (dummy_worker, expected_elements) in worker_info:
        with mock.patch.object(worker, "_worker_info", dummy_worker):
            wrapper = SimpleIterableStreamWrapper(stream=index_stream, shuffle=False)
            for _ in range(NUM_CYCLES):
                actual_elements = list(wrapper)
                test_results.append(
                    actual_elements == expected_elements
                    and len(actual_elements) == len(expected_elements)
                )
    assert all(test_results)
