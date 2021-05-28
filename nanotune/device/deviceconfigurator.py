import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Union

from qcodes.instrument.parameter import Parameter as QC_Parameter
from qcodes.monitor.monitor import Monitor
from qcodes.station import Station
from ruamel.yaml import YAML

import nanotune as nt
from nanotune.device.device import Device

logger = logging.getLogger(__name__)

msrmnt_prmtrs_tp = Dict[str, Union[None, QC_Parameter]]


class DeviceConfigurator:
    """"""

    def __init__(
        self,
        station: Optional[Station] = None,
        filename: str = nt.config["device_config_file"],
    ) -> None:

        self.monitor_parameters: Dict[int, Any] = {}

        if station is None:
            station = Station.default or Station()
        self.station = station

        self.filename = filename

        self.load_file()

    def load_file(self) -> None:
        """"""
        yaml = YAML()

        with open(self.filename, "r") as f:
            self.config = yaml.load(f)

    def load_device(self, name: str, **kwargs) -> Device:
        """"""
        if name not in self.config["devices"].keys():
            raise RuntimeError("Instrument {} not found in config.".format(name))
        device_params = self.config["devices"][name]

        init_kwargs = device_params.get("init", {})
        # somebody might have a empty init section in the config
        init_kwargs = {} if init_kwargs is None else init_kwargs

        device_kwargs = deepcopy(init_kwargs)
        device_kwargs.update(kwargs)

        # TO DO: check if device_kwargs have all necessary components
        # Turn strings to instruments and parameters where needed
        m_parameters = device_kwargs["measurement_parameters"]
        for readout_methods, meas_par in m_parameters.items():
            qc_prms: Optional[msrmnt_prmtrs_tp] = None
            # for meas_par in m_params:
            if meas_par is not None:
                if len(meas_par.split(".")) != 2:
                    logger.error("Unsupported measurement parameter.")
                    raise ValueError
                instr, prm = meas_par.split(".")
                # qc_prms.append(getattr(self.station.components[instr], prm))
                qc_prms = getattr(self.station.components[instr], prm)

            device_kwargs["measurement_parameters"][readout_methods] = qc_prms

        for ip, gate_pars in enumerate(device_kwargs["gate_parameters"]):
            instr = self.station.components[gate_pars[2]]
            device_kwargs["gate_parameters"][ip][2] = instr

        for ip, gate_pars in enumerate(device_kwargs["sensor_parameters"]):
            instr = self.station.components[gate_pars[2]]
            device_kwargs["sensor_parameters"][ip][2] = instr

        if device_kwargs["ohmic_parameters"] is not None:
            for ip, ohmic_pars in enumerate(device_kwargs["ohmic_parameters"]):
                if ohmic_pars is not None:
                    instr = self.station.components[ohmic_pars[2]]
                    device_kwargs["ohmic_parameters"][ip][2] = instr
        else:
            device_kwargs["ohmic_parameters"] = None

        # print(device_kwargs)
        device = Device(name=name, **device_kwargs)

        for gate in device.gates:
            self.monitor_parameters[id(gate.dc_voltage)] = gate.dc_voltage

        for ohmic in device.ohmics:
            self.monitor_parameters[id(ohmic.state)] = ohmic.state

        # add the instrument to the station
        self.station.add_component(device, device.name)

        # restart Monitor
        # Monitor(*self.monitor_parameters.values())

        return device
