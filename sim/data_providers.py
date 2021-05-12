#pylint: too-many-arguments, too-many-locals

import os
from typing import (
    List,
    Dict,
    Optional,
    Any,
)
import qcodes as qc
import xarray as xr
from scipy import interpolate
from sim.qcodes_utils import QcodesDbConfig


class Pin:
    def __init__(
        self,
        name: str,
    ) -> None:
        self._name = name
        self._value: float = 0.0

    def __repr__(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        """Name of the pin"""

        return self._name

    @property
    def value(self) -> float:
        """Gets the current value of the pin"""

        return self._value

    def get_value(self) -> float:
        """Gets the current value on the input pin.  Compatible with qcodes
        Parameter get_cmd argument.
        """

        return self._value


class StaticDataProvider:
    """Data provider that returns a constant value for all inputs."""

    def __init__(self, value: float) -> None:
        """Initializes a static data provider."""

        self._value = value
        self._xarray = xr.DataArray("0", dims="x", coords={"x":[1]})

    def __call__(self, *args) -> float:
        return self._value

    @property
    def value(self) -> float:
        """ """

        return self._value

    def raw_data(self) -> xr.DataArray:
        """Returns the raw data backing this provider as an
        xarray.DataArray
        """

        return self._xarray


class QcodesDataProvider:
    """Data provider that sources it's data from a 1D or 2D QCoDeS dataset."""

    def __init__(
        self,
        input_providers: List[Pin],
        db_path: str,
        exp_name: str,
        run_id: int,
        model_param_name: Optional[str] = None,
    ) -> None:
        """

        Args:
            input_providers: List of INPUT dependencies used to compute the
                output data. All objects in the list must support a .value
                property.
            db_path: Path to the qcodes database containing the source data.
            exp_name: Name of the qcodes experiment containing the source data.
            run_id: Run_id of the qcodes dataset containing the source data.
            model_param_name: Name of the qcodes dataset parameter that
                represents the output results. The input parameters will be
                based on the dependent parameters.
        """

        self._inputs = input_providers
        dataset_id = f"{db_path}.{exp_name}.{run_id}"

        if not os.path.isfile(db_path):
            raise FileNotFoundError(db_path)

        with QcodesDbConfig(db_path):

            dataset = qc.load_by_run_spec(
                experiment_name=exp_name,
                captured_run_id=run_id,
            )
            dataset_params = dataset.get_parameters()

            # if a model name was specified, make sure it is actually in the
            # qcodes dataset
            if model_param_name and model_param_name not in dataset.paramspecs:
                raise KeyError(
                    f"model_param_name '{model_param_name}' not found in " \
                        f"dataset {dataset_id}"
                    )

            # collect a list of all of the parameters using the dependencies
            # specified on the model param
            if not model_param_name:
                output_paramspec = dataset_params[-1]
            else:
                output_paramspec = dataset.paramspecs[model_param_name]
            param_names = output_paramspec.depends_on.split(", ")
            param_names.append(output_paramspec.name)

            # For now, we'll only support up to 2 input dimensions.
            # e.g.  y = f(x) or z = f(x,y)
            input_dimensions = len(param_names) - 1

            if input_dimensions == 0:
                raise RuntimeError(
                    "The dataset '{dataset_id}' cannot be used as a basis for \
                    the model because it does not specify any dependent \
                    parameters."
                )

            if input_dimensions > 2:
                raise RuntimeError(
                    f"{self.__class__.__name__} currently only supports 1 and \
                    2 input dimensions, but the specified dataset \
                    '{dataset_id}' requires {input_dimensions}"
                )

            # make sure the number of input providers matches the number of
            # model dimensions
            if len(input_providers) != input_dimensions:
                raise ValueError(
                    f"Invalid number of input_providers: dataset_params \
                    specified {len(dataset_params)} parameters, indicating a \
                    {input_dimensions}-dimensional model, but \
                    {len(input_providers)} input_providers were specified."
                )

            # Get the xarray_dataset from the qcodes dataset
            # Rename the axes names to the qcodes data label (if available)
            # instead of the original qcodes param name, because it is
            # (should be) a more friendly name
            # e.g.  Will rename 'mdac1_chan11_voltage' to 'LeftP'
            param_labels = [
                dataset.paramspecs[name].label if dataset.paramspecs[name].label
                else name for name in param_names
            ]
            rename_dict = dict(zip(param_names, param_labels))
            renamed_dataset = dataset.to_xarray_dataset().rename(rename_dict)  # type: ignore
            self._xarray_dataset = renamed_dataset

            # set up the interpolation
            inputs = [self._xarray_dataset[param] for param in param_labels]
            if input_dimensions == 1:
                self._interpolate = interpolate.interp1d(
                    *inputs,
                    bounds_error = False,
                    fill_value="extrapolate")
            else:
                self._interpolate = interpolate.interp2d(
                    *inputs,
                    bounds_error = False,
                )

    @property
    def value(self) -> float:
        """Looks up and returns the measurement result given the bound inputs,
        using interpolation
        """

        inputs = [inpt.value for inpt in self._inputs]
        result = self._interpolate(*inputs).item()
        return result

    @property
    def raw_data(self) -> xr.Dataset:
        """ Returns the full data model as an xarray.DataSet """
        return self._xarray_dataset
