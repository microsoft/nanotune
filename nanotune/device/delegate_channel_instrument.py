# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import importlib
from qcodes.instrument.delegate import DelegateInstrument
from qcodes.station import Station
from typing import (
    Any,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
    Type,
    Dict,
)
from qcodes import InstrumentChannel


class DelegateChannelInstrument(DelegateInstrument):
    """DelegateInstrument is an instrument driver with one or more
    parameters that connect to instrument parameters.

    Example usage in instrument YAML:

    .. code-block:: yaml

        field:
            type: qcodes.instrument.delegate.DelegateInstrument
            init:
            parameters:
                X:
                    - field_X.field
                ramp_rate:
                    - field_X.ramp_rate
            channels:
                gate_1: dac.ch01
            set_initial_values_on_load: true
            initial_values:
                ramp_rate: 0.02
            setters:
                X:
                    method: field_X.set_field
                    block: false
            units:
                X: T
                ramp_rate: T/min

    Args:
        name: Instrument name
        station: Station containing the real instrument that is used to get the endpoint
            parameters.
        parameters: A mapping from the name of a parameter to the sequence
            of source parameters that it points to.
        initial_values: Default values to set on the delegate instrument's
            parameters. Defaults to None (no initial values are specified or set).
        set_initial_values_on_load: Flag to set initial values when the
            instrument is loaded. Defaults to False.
        setters: Optional setter methods to use instead of calling the ``.set()``
            method on the endpoint parameters. Defaults to None.
        units: Optional units to set for parameters.
        metadata: Optional metadata to pass to instrument. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        station: Station,
        parameters: Optional[Union[Mapping[str, Sequence[str]], Mapping[str, str]]] = None,
        channels: Optional[Union[Mapping[str, Mapping[str, Any]], Mapping[str, str]]] = None,
        initial_values: Optional[Mapping[str, Any]] = None,
        set_initial_values_on_load: bool = False,
        setters: Optional[Mapping[str, MutableMapping[str, Any]]] = None,
        units: Optional[Mapping[str, str]] = None,
        metadata: Optional[Mapping[Any, Any]] = None,
    ):
        if parameters is None:
            parameters = {}

        super().__init__(
            name=name,
            station=station,
            parameters=parameters,
            initial_values=initial_values,
            set_initial_values_on_load=set_initial_values_on_load,
            setters=setters,
            units=units,
            metadata=metadata,
        )

        if channels is not None:
            self._create_and_add_channels(
                station=station,
                channels=channels,
            )

    def _create_and_add_channels(
        self,
        station: Station,
        channels: Mapping[str, Union[str, Mapping[str, Any]]],
    ) -> None:
        """Add channels to the instrument."""
        channel_wrapper = None
        chnnls_dict: Dict[str, Union[str, Mapping[str, Any]]] = dict(channels)
        channel_type_global = chnnls_dict.pop("type", None)
        if channel_type_global is not None and \
           not isinstance(channel_type_global, str):
            raise ValueError("Wrong channel type.")
        channel_wrapper_global = _get_channel_wrapper_class(
            channel_type_global
        )

        for channel_name, input_params in chnnls_dict.items():
            if isinstance(input_params, Mapping):
                input_params = dict(input_params)
                channel_type_individual = input_params.pop("type", None)
                channel_wrapper_individual = _get_channel_wrapper_class(
                    channel_type_individual
                )
                if channel_wrapper_individual is None:
                    channel_wrapper = channel_wrapper_global
                else:
                    channel_wrapper = channel_wrapper_individual
            else:
                channel_wrapper = channel_wrapper_global

            self._create_and_add_channel(
                channel_name=channel_name,
                station=station,
                input_params=input_params,
                channel_wrapper=channel_wrapper,
            )

    def _create_and_add_channel(
        self,
        channel_name: str,
        station: Station,
        input_params: Union[str, Mapping[str, Any]],
        channel_wrapper: Optional[Type[InstrumentChannel]],
        **kwargs: Any,
    ) -> None:
        """Adds a channel to the instrument."""
        if isinstance(input_params, str):
            try:
                channel = self.parse_instrument_path(station, input_params)
            except ValueError as v_err:
                msg = "Unknown channel path."
                raise ValueError(msg) from v_err

        elif isinstance(input_params, Mapping) and channel_wrapper is not None:
            channel = self.parse_instrument_path(
                station, input_params["channel"]
            )
            wrapper_kwargs = dict(**kwargs, **input_params)

            channel = channel_wrapper(
                parent=channel.parent,
                name=channel_name,
                **wrapper_kwargs
            )
        else:
            raise ValueError(
                "Channels can only be created from existing channels, "
                "or using a wrapper channel class; "
                f"instead got {input_params!r} inputs with "
                f"{channel_wrapper!r} channel wrapper."
            )

        self.add_submodule(channel_name, channel)


    def __repr__(self) -> str:
        params = ", ".join(self.parameters.keys())
        return f"DelegateInstrument(name={self.name}, parameters={params})"

def _get_channel_wrapper_class(
    channel_type: Optional[str],
) -> Optional[Type[InstrumentChannel]]:
    """Get channel class from string specified in yaml."""
    if channel_type is None:
        return None
    else:
        channel_type_elems = str(channel_type).split(".")
        module_name = ".".join(channel_type_elems[:-1])
        instr_class_name = channel_type_elems[-1]
        module = importlib.import_module(module_name)
        return getattr(module, instr_class_name)
