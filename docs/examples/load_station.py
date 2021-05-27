# Load station
import qcodes as qc
from mocks import MockDAC, MockLockin, MockReadoutInstrument, MockRF, MockField

dac = MockDAC('dac', num_channels=3)

field_X = MockField(
    name='field_X'
)

rf_source1 = MockRF(
    name='rf_source1'
)

rf_source2 = MockRF(
    name='rf_source2'
)

dmm = MockReadoutInstrument(
    name='dmm',
    setter_param=dac.ch01.voltage,
    rf_source=rf_source1,
    field=field_X.field,
    parameter_name="volt"
)

lockin = MockLockin(
    name='lockin',
    setter_param=dac.ch01.voltage,
    field=field_X.field
)

station = qc.Station()
station.add_component(dac)
station.add_component(dmm)
station.add_component(lockin)
station.add_component(rf_source1)
station.add_component(rf_source2)
station.add_component(field_X)

station.load_config_file("chip.yaml")
chip = station.load_MockChip_123(station=station)
field = station.load_field(station=station)
mux = station.load_mux(station=station)
