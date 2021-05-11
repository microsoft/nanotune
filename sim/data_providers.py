#pylint: disable=line-too-long, too-many-arguments, too-many-locals

""" Contains classes that can be used to simulate physical devices using a variety of data providers as a basis for a simulation model """

import os
import qcodes as qc
import xarray as xr
from scipy import interpolate
from sim.qcodes_utils import QcodesDbConfig


class StaticDataProvider:

    """ Data provider that returns a constant value for all inputs."""

    def __init__(self, value):
        self._value = value
        self._xarray = xr.DataArray("0", dims="x", coords={"x":[1]})

    def __call__(self, *args):
        return self._value

    @property
    def value(self):
        return self._value

    def raw_data(self):
        """ Returns the raw data backing this provider as an xarray.DataArray """
        return self._xarray


class QcodesDataProvider:

    """ Data provider that sources it's data from a 1D or 2D QCoDeS dataset."""

    def __init__(self, input_providers : list, db_path : str, exp_name : str, run_id : int, model_param_name : str = None):
        """ Parameters:
              input_providers   List of INPUT dependencies used to compute the output data.
                                All objects in the list must support a .value property

              db_path           Path to the qcodes database containing the source data
              exp_name          Name of the qcodes experiment containing the source data
              run_id            Run_id of the qcodes dataset containing the source data

              model_param_name  Name of the qcodes dataset parameter that represents the output results.
                                The input parameters will be based on the dependent parameters.
        """

        self._inputs = input_providers
        dataset_id = f"{db_path}.{exp_name}.{run_id}"

        if not os.path.isfile(db_path):
            raise FileNotFoundError(db_path)

        with QcodesDbConfig(db_path):

            dataset = qc.load_by_run_spec(experiment_name=exp_name, captured_run_id=run_id)
            dataset_params = dataset.get_parameters()

            # if a model name was specified, make sure it is actually in the qcodes dataset
            if model_param_name and model_param_name not in dataset.paramspecs:
                raise KeyError(f"model_param_name '{model_param_name}' not found in dataset {dataset_id}")

            # collect a list of all of the parameters using the dependencies specified on the model param
            output_paramspec = dataset_params[-1] if not model_param_name else dataset.paramspecs[model_param_name]
            param_names = output_paramspec.depends_on.split(", ")
            param_names.append(output_paramspec.name)

            # For now, we'll only support up to 2 input dimensions.  e.g.  y = f(x) or z = f(x,y)
            input_dimensions = len(param_names) - 1

            if input_dimensions == 0:
                raise RuntimeError("The dataset '{dataset_id}' cannot be used as a basis for the model because it does not specify any dependent parameters.")

            if input_dimensions > 2:
                raise RuntimeError(f"{self.__class__.__name__} currently only supports 1 and 2 input dimensions, but the specified dataset '{dataset_id}' requires {input_dimensions}")

            # make sure the number of input providers matches the number of model dimensions
            if len(input_providers) != input_dimensions:
                raise ValueError(f"Invalid number of input_providers: dataset_params specified {len(dataset_params)} parameters, indicating a {input_dimensions}-dimensional model, but {len(input_providers)} input_providers were specified.")

            # Get the xarray_dataset from the qcodes dataset
            # Rename the axes names to the qcodes data label (if available) instead of the original qcodes param name, because it is (should be) a more friendly name
            # e.g.  Will rename 'mdac1_chan11_voltage' to 'LeftP'
            param_labels = [dataset.paramspecs[name].label if dataset.paramspecs[name].label else name for name in param_names]
            rename_dict = dict(zip(param_names, param_labels))
            self._xarray_dataset = dataset.to_xarray_dataset().rename(rename_dict)

            # set up the interpolation
            inputs = [self._xarray_dataset[param] for param in param_labels]
            self._interpolate = interpolate.interp1d(*inputs, bounds_error = False, fill_value="extrapolate") if input_dimensions == 1 else interpolate.interp2d(*inputs, bounds_error = False)


    @property
    def value(self):
        """ Looks up and returns the measurement result given the bound inputs, using interpolation """
        inputs = [input.value for input in self._inputs]
        result = self._interpolate(*inputs).item()
        return result

    @property
    def raw_data(self):
        """ Returns the full data model as an xarray.DataSet """
        return self._xarray_dataset
