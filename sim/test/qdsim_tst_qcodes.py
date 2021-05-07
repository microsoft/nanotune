import os
import numpy as np
import qcodes as qc
from qcodes_mocks import QuantumDotMockInstrument
from data_providers import QcodesDataProvider


char_db_path = os.path.join(os.getcwd(), "data", "QuantumDots", "device_characterization.db")
tune_db_path = os.path.join(os.getcwd(), "data", "QuantumDots", "dot_tuning_sequences.db")

qdMockInstr = QuantumDotMockInstrument()
qdsim = qdMockInstr.simulator

exp = qc.load_or_create_experiment("simtest")
station = qc.Station()
station.add_component(qdMockInstr)

# setup the simulator with the correct data provider
pinchoff_lp = QcodesDataProvider([qdsim.l_plunger], char_db_path, "GB_Newtown_Dev_3_2", 1204)
qdsim.drain.set_data_provider(pinchoff_lp)

# run the measurement
meas = qc.Measurement(exp = exp, station = station)
meas.register_parameter(qdMockInstr.left_plunger)
meas.register_parameter(qdMockInstr.drain, setpoints=(qdMockInstr.left_plunger,))
meas.write_period = 2

readings = 43

with meas.run() as datasaver:
    for voltage in np.linspace(0.0, -0.8, readings):
        qdMockInstr.left_plunger.set(voltage)
        current = qdMockInstr.drain.get()
        datasaver.add_result((qdMockInstr.left_plunger, voltage), (qdMockInstr.drain, current))

    dataset = datasaver.dataset

print("done")