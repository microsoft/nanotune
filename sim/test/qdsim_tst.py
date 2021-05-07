from qdsim import QcodesDataProvider, QuantumDotSim, dump_db
import qcodes as qc
import numpy as np
import xarray as xr
import os
import qcodes as qc
from qcodes.tests.instrument_mocks import DummyInstrument
# -----------------
# playground
# -----------------


# Always start with QCoDeS pointing to my default experiments DB and verify at the end that it still is.
original_db_path = "c:\\users\\jalee\experiments.db"
qc.config["core"]["db_location"] = original_db_path

# Setup a valid simulator and sweep with each of the emulation modes
char_db_path = os.path.join(os.getcwd(), "data", "QuantumDots", "device_characterization.db")
tune_db_path = os.path.join(os.getcwd(), "data", "QuantumDots", "dot_tuning_sequences.db")
#dump_db(pinchoff_db_path)
#dump_db(chargestate_db_path)


qdsim = QuantumDotSim()
print(qdsim)
qdp = QcodesDataProvider( input_providers = [qdsim.lp, qdsim.rp], db_path=tune_db_path, exp_name="GB_Newtown_Dev_1_1", run_id = 19, model_param_name = "" )



pinchoff_lb = DataProvider_Qcodes1D([qdsim.lb], char_db_path, "GB_Newtown_Dev_3_2", 1203)
pinchoff_lp = DataProvider_Qcodes1D([qdsim.lp], char_db_path, "GB_Newtown_Dev_3_2", 1204)
pinchoff_cb = DataProvider_Qcodes1D([qdsim.cb], char_db_path, "GB_Newtown_Dev_3_2", 1205)
pinchoff_rp = DataProvider_Qcodes1D([qdsim.rp], char_db_path, "GB_Newtown_Dev_3_2", 1206)
pinchoff_rb = DataProvider_Qcodes1D([qdsim.rb], char_db_path, "GB_Newtown_Dev_3_2", 1207)

charge_state_diagram = DataProvider_Qcodes2D(inputs = [qdsim.lp, qdsim.rp], db_path=tune_db_path, exp_name="GB_Newtown_Dev_1_1", run_id = 19)


# Set the data provider for the left barrier pinchoff sweep
qdsim.drain.set_data_provider(pinchoff_lb)

# Helper to set the value on the input pin, then return the value seen at the output pin
def set_input_then_read_output(x, input_pin, output_pin):
    input_pin.set_value(x)
    return output_pin.value
