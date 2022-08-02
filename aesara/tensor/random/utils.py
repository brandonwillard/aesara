from collections.abc import Sequence
from functools import wraps
from itertools import zip_longest
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from typing_extensions import Literal

from aesara.compile.sharedvalue import shared
from aesara.graph.basic import Constant, Variable
from aesara.tensor import get_vector_length
from aesara.tensor.basic import as_tensor_variable, cast, constant
from aesara.tensor.extra_ops import broadcast_to
from aesara.tensor.math import maximum
from aesara.tensor.shape import specify_shape
from aesara.tensor.type import int_dtypes
from aesara.tensor.var import TensorVariable


if TYPE_CHECKING:
    from aesara.tensor.random.op import RandomVariable


def params_broadcast_shapes(param_shapes, ndims_params, use_aesara=True):
    """Broadcast parameters that have different dimensions.

    Parameters
    ==========
    param_shapes : list of ndarray or Variable
        The shapes of each parameters to broadcast.
    ndims_params : list of int
        The expected number of dimensions for each element in `params`.
    use_aesara : bool
        If ``True``, use Aesara `Op`; otherwise, use NumPy.

    Returns
    =======
    bcast_shapes : list of ndarray
        The broadcasted values of `params`.
    """
    max_fn = maximum if use_aesara else max

    rev_extra_dims = []
    for ndim_param, param_shape in zip(ndims_params, param_shapes):
        # We need this in order to use `len`
        param_shape = tuple(param_shape)
        extras = tuple(param_shape[: (len(param_shape) - ndim_param)])

        def max_bcast(x, y):
            if getattr(x, "value", x) == 1:
                return y
            if getattr(y, "value", y) == 1:
                return x
            return max_fn(x, y)

        rev_extra_dims = [
            max_bcast(a, b)
            for a, b in zip_longest(reversed(extras), rev_extra_dims, fillvalue=1)
        ]

    extra_dims = tuple(reversed(rev_extra_dims))

    bcast_shapes = [
        (extra_dims + tuple(param_shape)[-ndim_param:])
        if ndim_param > 0
        else extra_dims
        for ndim_param, param_shape in zip(ndims_params, param_shapes)
    ]

    return bcast_shapes


def broadcast_params(params, ndims_params):
    """Broadcast parameters that have different dimensions.

    >>> ndims_params = [1, 2]
    >>> mean = np.array([1, 2, 3])
    >>> cov = np.stack([np.eye(3), np.eye(3)])
    >>> params = [mean, cov]
    >>> res = broadcast_params(params, ndims_params)
    [array([[1, 2, 3]]),
    array([[[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]],
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]])]

    Parameters
    ==========
    params : list of ndarray
        The parameters to broadcast.
    ndims_params : list of int
        The expected number of dimensions for each element in `params`.

    Returns
    =======
    bcast_params : list of ndarray
        The broadcasted values of `params`.
    """
    use_aesara = False
    param_shapes = []
    for p in params:
        param_shape = tuple(
            1 if bcast else s
            for s, bcast in zip(p.shape, getattr(p, "broadcastable", (False,) * p.ndim))
        )
        use_aesara |= isinstance(p, Variable)
        param_shapes.append(param_shape)

    shapes = params_broadcast_shapes(param_shapes, ndims_params, use_aesara=use_aesara)
    broadcast_to_fn = broadcast_to if use_aesara else np.broadcast_to

    bcast_params = [
        broadcast_to_fn(param, shape) for shape, param in zip(shapes, params)
    ]

    return bcast_params


def normalize_size_param(
    size: Optional[Union[int, np.ndarray, Variable, Sequence]]
) -> Variable:
    """Create an Aesara value for a ``RandomVariable`` ``size`` parameter."""
    if size is None:
        size = constant([], dtype="int64")
    elif isinstance(size, int):
        size = as_tensor_variable([size], ndim=1)
    elif not isinstance(size, (np.ndarray, Variable, Sequence)):
        raise TypeError(
            "Parameter size must be None, an integer, or a sequence with integers."
        )
    else:
        size = cast(as_tensor_variable(size, ndim=1), "int64")

        if not isinstance(size, Constant):
            # This should help ensure that the length of non-constant `size`s
            # will be available after certain types of cloning (e.g. the kind
            # `Scan` performs)
            size = specify_shape(size, (get_vector_length(size),))

    assert not any(s is None for s in size.type.shape)
    assert size.dtype in int_dtypes

    return size


class BaseRandomStream:
    r"""Creates an interface similar to `numpy.random.Generator` for `RandomType`\s.

    Attributes
    ----------
    seed: None or int
        A default seed to initialize the `Generator` instances after build.
    default_instance_seed: int
        Instance variable should take None or integer value. Used to seed the
        random number generator that provides seeds for member streams.
    gen_seedgen: numpy.random.Generator
        `Generator` instance that `RandomStream.gen` uses to seed new
        streams.
    symbolic_rng_ctor: type
        Constructor used to create the underlying symbolic RNG objects.

    """

    def __init__(
        self,
        seed: Optional[int],
        namespace: Optional[ModuleType],
        symbolic_rng_ctor: Callable[[Optional[int]], Variable],
    ):
        if namespace is None:
            from aesara.tensor.random import basic  # pylint: disable=import-self

            self.namespaces = [basic]
        else:
            self.namespaces = [namespace]

        self.default_instance_seed = seed
        self.state_updates = []
        self.gen_seedgen = np.random.SeedSequence(seed)
        self.symbolic_rng_ctor = symbolic_rng_ctor

    def __getattr__(self, obj):

        ns_obj = next(
            (getattr(ns, obj) for ns in self.namespaces if hasattr(ns, obj)), None
        )

        if ns_obj is None:
            raise AttributeError("No attribute {}.".format(obj))

        from aesara.tensor.random.op import RandomVariable

        if isinstance(ns_obj, RandomVariable):

            @wraps(ns_obj)
            def meta_obj(*args, **kwargs):
                return self.gen(ns_obj, *args, **kwargs)

        else:
            raise AttributeError("No attribute {}.".format(obj))

        setattr(self, obj, meta_obj)
        return getattr(self, obj)

    def seed(self, seed: Optional[int] = None) -> None:
        """Reset the internal seed-generating process.

        Parameters
        ----------
        seed
            Each random stream will be assigned a unique state that depends
            deterministically on this value.

        """
        if seed is None:
            seed = self.default_instance_seed

        self.gen_seedgen = np.random.SeedSequence(seed)

    def gen(self, op: "RandomVariable", *args, **kwargs) -> TensorVariable:
        r"""Generate a draw from `op` seeded by this `RandomStream`.

        Parameters
        ----------
        op
            A `RandomVariable` instance
        args
            Positional arguments passed to `op`.
        kwargs
            Keyword arguments passed to `op`.

        Returns
        -------
        The symbolic random draw performed by `op`.  This function stores
        the updated `RandomType`\s for use at compile time.

        """
        if "rng" in kwargs:
            raise ValueError(
                "The `rng` option cannot be used with a variate in a `RandomStream`"
            )

        # Generate a new random state
        (seed,) = self.gen_seedgen.spawn(1)

        rng = self.symbolic_rng_ctor(seed)

        # Generate the sample
        out = op(*args, **kwargs, rng=rng)

        return out


class SharedRandomStream(BaseRandomStream):
    r"""Creates an interface similar to `numpy.random.Generator` for shared `RandomType`\s.

    Attributes
    ----------
    state_updates: list
        A list of pairs of the form ``(input_r, output_r)``.  This will be
        over-ridden by the module instance to contain stream generators.
    numpy_rng_ctor: type
        Constructor used to create the underlying NumPy RNG objects.  The
        default is `np.random.default_rng`.

    """

    def __init__(
        self,
        seed: Optional[int] = None,
        namespace: Optional[ModuleType] = None,
        symbolic_rng_ctor: Optional[Callable[[Optional[int]], Variable]] = None,
        rng_ctor: Literal[
            np.random.RandomState, np.random.Generator
        ] = np.random.default_rng,
    ):
        self.state_updates = []

        if isinstance(rng_ctor, type) and issubclass(rng_ctor, np.random.RandomState):

            # The legacy state does not accept `SeedSequence`s directly
            def rng_ctor(seed):
                return np.random.RandomState(np.random.MT19937(seed))

        self.rng_ctor = rng_ctor

        if symbolic_rng_ctor is None:

            def symbolic_rng_ctor(seed):
                return shared(self.rng_ctor(seed), borrow=True)

        super().__init__(seed, namespace, symbolic_rng_ctor)

    def updates(self):
        return list(self.state_updates)

    def seed(self, seed: Optional[None] = None) -> None:
        """Re-initialize each random stream.

        Parameters
        ----------
        seed
            Each random stream will be assigned a unique state that depends
            deterministically on this value.

        """
        super().seed(seed)

        old_r_seeds = self.gen_seedgen.spawn(len(self.state_updates))

        for (old_r, new_r), old_r_seed in zip(self.state_updates, old_r_seeds):
            old_r.set_value(self.rng_ctor(old_r_seed), borrow=True)

    def gen(self, op: "RandomVariable", *args, **kwargs) -> TensorVariable:
        r"""Generate a draw from `op` seeded from this `RandomStream`.

        Parameters
        ----------
        op
            A `RandomVariable` instance
        args
            Positional arguments passed to `op`.
        kwargs
            Keyword arguments passed to `op`.

        Returns
        -------
        The symbolic random draw performed by `op`.  This function stores
        the updated `RandomType`\s for use at compile time.

        """

        out = super().gen(op, *args, **kwargs)

        rng = out.owner.inputs[0]

        # This is the value that should be used to replace the old state
        # (i.e. `rng`) after `out` is sampled/evaluated.
        # The updates mechanism in `aesara.function` is supposed to perform
        # this replace action.
        new_rng = out.owner.outputs[0]

        self.state_updates.append((rng, new_rng))

        rng.default_update = new_rng

        return out


# For backward compatibility
RandomStream = SharedRandomStream


def default_rng(seed: Optional[int] = None, shared: bool = True) -> BaseRandomStream:
    """Construct a new Generator with the default BitGenerator (PCG64).

    Parameters
    ----------
    seed
        The initial seed value.
    shared
        If ``True``, create shared variable RNG objects.

    """
    if shared:
        return SharedRandomStream(seed)
    else:
        from aesara.tensor.random.op import default_rng

        return BaseRandomStream(seed, None, default_rng)
