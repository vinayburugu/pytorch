from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from torch.fx import Node
from typing import Callable, List, NamedTuple, Optional, Dict, Any
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor

import torch

__all__ = [
    "Quantizer",
    "QuantizationSpec",
    "QuantizationAnnotation",
]

# TODO: maybe remove torch.float32
SUPPORTED_DTYPES = [torch.uint8, torch.int8, torch.int32, torch.float16, torch.float32]
SUPPORTED_QSCHEMES = [
    torch.per_tensor_affine,
    torch.per_tensor_symmetric,
    torch.per_channel_affine,
    torch.per_channel_symmetric,
    torch.per_channel_affine_float_qparams,
]

@dataclass(eq=True, frozen=True)
class QuantizationSpec:
    dtype: torch.dtype
    # observer or fake_quantize constructor such as
    # MinMaxObserver, PerChannelHistogramObserver etc.
    # or we can attach some custom args to them
    # e.g. MinMaxObserver.with_args(eps=eps)
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[torch.qscheme] = None
    ch_axis: Optional[int] = None
    is_dynamic: bool = False

    def __post_init__(self):
        # check dtype is one of the supported types
        if self.dtype not in SUPPORTED_DTYPES:
            raise TypeError(f"Unsupported dtype {self.dtype}.")

        # quant_min must be less than quant_max
        if (
            self.quant_min is not None
            and self.quant_max is not None
            and self.quant_min > self.quant_max
        ):
            raise ValueError(
                f"quant_min {self.quant_min} must be <= quant_max {self.quant_max}."
            )

        # check qscheme is on of the supported ones
        if self.qscheme is not None and self.qscheme not in SUPPORTED_QSCHEMES:
            raise ValueError(f"Unsupported qscheme {self.qscheme}.")

        # ch_axis must be less than the number of channels
        # but no way to check here. Just check that it is not < 0.
        if self.ch_axis is not None and self.ch_axis < 0:
            raise ValueError("Ch_axis is < 0.")


# In the absence of better name, just winging it with QuantizationConfig
@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec]
    # TODO: remove, since we can use observer_or_fake_quant_ctr to express this
    is_qat: bool = False

OperatorPatternType = List[Callable]

OperatorConfig = NamedTuple(
    "OperatorConfig",
    # fix List[str] with List[List[Union[nn.Module, FunctionType, BuiltinFunctionType]]]
    # Basically we are mapping a quantization config to some list of patterns.
    # a pattern is defined as a list of nn module, function or builtin function names
    # e.g. [nn.Conv2d, torch.relu, torch.add]
    # We have not resolved whether fusion can be considered internal details of the
    # quantizer hence it does not need communication to user.
    # Note this pattern is not really informative since it does not really
    # tell us the graph structure resulting from the list of ops.
    [
        ("config", QuantizationConfig),
        (
            "operators",
            List[OperatorPatternType],
        ),
    ],
)

@dataclass
class QuantizationAnnotation:
    """ How are input arguemnt or output should be quantized,
    expressed as QuantizationSpec, this corresponds to how a Tensor in the
    operator Graph is observed (PTQ) or fake quantized (QAT)
    """

    # a map from torch.fx.Node to QuantizationSpec
    # TODO: change the value to QuantizationSpec in a separate PR
    input_qspec_map: Dict[Node, Any] = field(default_factory=dict)

    # How the output of this node is quantized, expressed as QuantizationSPec
    # TODO: change the value to QuantizationSpec in a separate PR
    output_qspec: Optional[Any] = None

    # whether the node is annotated or not
    _annotated: bool = False

    # TODO: will be updated soon to use sharing group and be more general
    _input_output_share_observers: bool = False

    # TODO: remove after sharing API refactor
    _reuse_input_obs_or_fq: bool = False

class Quantizer(ABC):

    # annotate nodes in the graph with observer or fake quant constructors
    # to convey the desired way of quantization
    @abstractmethod
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        pass

    # validate the annotated graph is supported by the backend
    @abstractmethod
    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    # annotate nodes in the graph with observer or fake quant constructors
    # to convey the desired way of quantization
    @classmethod
    @abstractmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        pass
