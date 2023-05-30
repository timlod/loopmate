from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Union

# import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig

from loopmate.utils import (
    EMA_MinMaxTracker,
    PeakTracker,
    SharedInt,
    magsquared,
    samples_to_frames,
    tempo_frequencies,
)


def query_circular(
    data: np.ndarray,
    idx_slice: slice,
    counter: int,
    out: Optional[np.ndarray] = None,
    axis: int = 0,
):
    """Return n samples, backwards from current counter.  Note: returns a copy
    of the requested data

    :param data: array to make a circular query into
    :param idx_slice: slice of samples to return.  Needs to satisfy -self.N <
        start < stop <= 0.  Ignores slice step.
    :param counter: index pointing to the latest entry in data (counter + 1
        will be the last entry)
    :param out: array to place the samples into.  Can be used to re-use an
        array of loop_length for sample storage to avoid extra memory copies.
    :param axis: either 0 (slice first axis) or -1 (slice last axis)
    """
    assert isinstance(
        idx_slice, slice
    ), f"Use slice for indexing! (Got {idx_slice})"
    start, stop = idx_slice.start or 0, idx_slice.stop or 0
    N = data.shape[axis]
    assert (
        -N <= start < stop <= 0
    ), f"Can only slice at most N ({N}) items backward on! {start}/{stop}"
    l_i = counter + start
    r_i = counter + stop

    if l_i < 0 <= r_i:
        if axis != 0:
            return np.concatenate(
                (data[..., l_i:], data[..., :r_i]), out=out, axis=axis
            )
        else:
            return np.concatenate((data[l_i:], data[:r_i]), out=out, axis=axis)
    else:
        if out is not None:
            if axis != 0:
                out[:] = data[..., l_i:r_i]
            else:
                out[:] = data[l_i:r_i]
            return out
        else:
            if axis != 0:
                return data[..., l_i:r_i].copy()
            else:
                return data[l_i:r_i].copy()


class CircularArray:
    """
    Simple implementation of an array which can be indexed and written to in a
    wrap-around fashion.
    """

    def __init__(self, data: np.ndarray, write_counter=0, counter=0, axis=0):
        """
        Initialize CircularArray given numpy array and counters.  Can be backed
        by shared memory.

        :param data: numpy array
        :param write_counter: can be wrapped in a SharedInt
        :param counter: can be wrapped in a SharedInt
        :param axis: axis along which to wrap the array.  Needs to be first or
            last axis (0 or -1)!
        """
        self.data = data
        assert axis in (0, -1), "Axis needs to be either 0 or -1!"
        self.axis = axis
        self.N = data.shape[axis]
        self.write_counter = write_counter
        self.counter = counter

    def query(self, i: slice | int, out=None):
        """Return n samples.  Note: returns a copy of the requested data
        (unless we specify the output array, in which case it writes a copy
        into it)!

        :param i: index or slice of samples to return.  Needs to satisfy
                  -self.N < start < stop <= 0.  Ignores slice step.
        :param out: array to place the samples into.  Can be used to re-use an
            array of loop_length for sample storage to avoid extra memory
            copies.
        """
        if isinstance(i, int):
            if self.axis == 0:
                return self.data[self.index_offset(i)]
            else:
                return self.data[..., self.index_offset(i)]
        return query_circular(
            self.data, i, int(self.write_counter), out, axis=self.axis
        )

    def __getitem__(self, i):
        """Get samples from this array. This returns a copy.

        :param i: slice satisfying -self.N < start < stop <= 0. Can't use step.
        """
        return self.query(i)

    def index_offset(self, offset: Union[int, np.ndarray]):
        i = self.write_counter + offset
        if isinstance(offset, np.ndarray):
            return np.where(
                i > self.N,
                i % self.N,
                np.where(i < 0, self.N + i, i),
            )
        else:
            if i > self.N:
                return i % self.N
            elif i < 0:
                return self.N + i
            else:
                return i

    def elements_since(self, c0):
        return self.counter - c0

    def write(self, arr, increment=True):
        """Write to this circular array.

        :param arr: array to write
        :param increment: whether to increment counters.  Use False only if
            counters are shared and should only be incremented once each
            timestep!
        """
        n = arr.shape[self.axis]
        arr_i = 0

        l_i = 0 + self.write_counter
        self.write_counter += n
        self.counter += n
        if self.write_counter >= self.N:
            arr_i = self.N - l_i
            if self.axis == 0:
                self.data[l_i:] = arr[:arr_i]
            elif self.axis == -1:
                self.data[..., l_i:] = arr[..., :arr_i]
            self.write_counter %= self.N
            l_i = 0
        if self.axis == 0:
            self.data[l_i : self.write_counter] = arr[arr_i:]
        else:
            self.data[..., l_i : self.write_counter] = arr[..., arr_i:]

    def __repr__(self):
        return (
            self.data.__repr__()
            + f"\nN: {self.N}, i: {self.write_counter}, c: {self.counter}"
        )
