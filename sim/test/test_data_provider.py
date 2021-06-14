import os
from itertools import product
from math import isclose

import pytest
import time

import sim
from sim.data_providers import (
    DelayedDataProvider,
    PassthroughDataProvider,
    QcodesDataProvider,
    RampedValueDataProvider,
    StaticDataProvider,
    SyntheticPinchoffDataProvider )
from sim.mock_devices import MockFieldWithRamp, Pin
from sim.simulation_scenario import SimulationScenario

testroot = os.path.dirname(os.path.abspath(__file__))
simroot = os.path.dirname(os.path.dirname(os.path.abspath(sim.__file__)))
scenario_root = os.path.join(testroot, "scenarios")

valid_db_path = os.path.join(
    simroot,
    "data",
    "tuning",
    "dot_tuning_sequences.db",
)

class TestQcodesDataProvier:
    def test_valid_1d_data_provider(self):
        """Create a 1D Data provider, and verify that reading from it with
        different input values yields different outputs.
        """

        in1 = Pin("in1")
        data_1d = QcodesDataProvider(
            [in1], valid_db_path, "GB_Newtown_Dev_3_2", 986
        )

        in1.set_value(0.0)
        out0 = data_1d.get_value()

        in1.set_value(-0.5)
        out1 = data_1d.get_value()

        assert out0 != out1

    def test_valid_2d_data_provider(self):
        """Create a 2D Data provider, and verify that reading from it with
        different input values yields different outputs.
        """

        in1 = Pin("in1")
        in2 = Pin("in2")
        data_2d = QcodesDataProvider(
            [in1, in2], valid_db_path, "GB_Newtown_Dev_3_2", 991
        )

        in1.set_value(0.0)
        in2.set_value(0.0)
        out0 = data_2d.get_value()

        in1.set_value(-0.5)
        in2.set_value(-0.5)
        out1 = data_2d.get_value()
        assert out0 != out1


    def test_bad_db_filename(self):
        """Validate the correct error is raised when trying to load a bad
        database.
        """

        with pytest.raises(FileNotFoundError):
            data_1d = QcodesDataProvider(
                [], "bad_database.db", "GB_Newtown_Dev_3_2", 991
            )

    def test_dataset_not_found(self):
        """Validates the correct error is raised if the specified dataset is
        not found in the db.
        """
        with pytest.raises(NameError):
            data_1d = QcodesDataProvider(
                [Pin("in1")],
                valid_db_path,
                "Bad_Experiment_Name",
                986,
            )

        with pytest.raises(NameError):
            data_1d = QcodesDataProvider(
                [Pin("in1")],
                valid_db_path,
                "GB_Newtown_Dev_3_2",
                987654321,
            )

    def test_good_dataset_parameter(self):
        """Validates the data provider uses the specified output parameter from
        the source dataset.
        """

        data_1d = QcodesDataProvider(
            [Pin("in1")],
            valid_db_path,
            "GB_Newtown_Dev_3_2",
            986,
            model_param_name="sr860_2_R_current",
        )

        assert data_1d.get_value() != 0.0

    def test_bad_dataset_parameter(self):
        """Validates that the correct error is raised if the specified parameter
        name is not found in the dataset.
        """

        with pytest.raises(KeyError):
            data_1d = QcodesDataProvider(
                [Pin("in1")],
                valid_db_path,
                "GB_Newtown_Dev_3_2",
                986,
                model_param_name="BAD_PARAM_NAME",
            )

    def test_too_few_input_bindings(self):
        """Validates that the correct error is raised if too few input binds are
        specified for indexing into the specified dataset.
        """

        in1 = Pin("in1")

        with pytest.raises(ValueError):
            # This is a 2D data set, but only 1 input binding is specified
            data_1d = QcodesDataProvider(
                [], valid_db_path, "GB_Newtown_Dev_3_2", 986
            )

        with pytest.raises(ValueError):
            # This is a 2D data set, but only 1 input binding is specified
            data_2d = QcodesDataProvider(
                [in1], valid_db_path, "GB_Newtown_Dev_3_2", 991
            )

    def test_too_many_input_bindings(self):
        """Validates that the correct error is raised if too many input binds
        are specified for indexing into the specified dataset.
        """

        in1 = Pin("in1")
        in2 = Pin("in2")
        in3 = Pin("in3")

        with pytest.raises(ValueError):
            # This is a 2D data set, but only 1 input binding is specified
            data_1d = QcodesDataProvider(
                [in1, in2], valid_db_path, "GB_Newtown_Dev_3_2", 986
            )

        with pytest.raises(ValueError):
            # This is a 2D data set, but only 1 input binding is specified
            data_2d = QcodesDataProvider(
                [in1, in2, in3], valid_db_path, "GB_Newtown_Dev_3_2", 991
            )


class TestStaticDataProvider:

    def test_static_data_provider(self):
        o1 = Pin("O1")
        o1.set_data_provider(StaticDataProvider(3.14))

        assert o1.get_value() == 3.14


class TestPassthroughDataProvider:

    def test_passthrough_data_provider(self):

        def assert_the_same(pin1 : Pin, pin2 : Pin, value: float) -> None:
            assert(pin1.get_value() == value)
            assert(pin2.get_value() == value)

        pin1 = Pin("Pin1")
        pin2 = Pin("Pin2")

        pin2.set_data_provider(PassthroughDataProvider(pin1))
        assert_the_same(pin1, pin2, 0.0)

        pin1.set_value(1.0)
        assert_the_same(pin1, pin2, 1.0)

        pin2.set_value(2.0)
        assert_the_same(pin1, pin2, 2.0)

class TestSyntheticPinchoffDataProvider:


    def test_pinchoff(self):
        """ Tests a variety of pinchoff curves occuring in each cartesian quadrant
            with varying heights and widths, and both flipped and non-flipped """

        pin = Pin("pin")
        centers = [-2.5, -1.0, 0.0, 1.0, 2.5]
        widths = [0.25, 0.5, 1.0, 2.0]
        mins = [-4.0, -2.5, -1.0, 0.0, 1.0, 2.5, 4.0]
        maxs = [-3.0, -1.5, -0.5, 0.0, 0.5, 3.5, 5.0]
        flips = [False, True]

        tolerance = 1e-6
        for center, width, min, max, flip in product(centers, widths, mins, maxs, flips):
            po = SyntheticPinchoffDataProvider(
                pin,
                min = min, max = max, center = center, width = width, flip = flip
            )

            if (min < max):
                config = f"min={min}, max={max}, center={center}, width={width}, flip={flip}"

                # Check expected center point value
                mid = min + (max-min)/2
                actual = po.compute(center)
                assert isclose(mid, actual, abs_tol = tolerance), f"Expected value at center={mid}, Actual={actual}.  {config}"

                # Check curve goes to expected bound left of center
                x_left = center - 1.5*width
                y_left = po.compute(x_left)
                expected_left = max if flip else min
                assert isclose(expected_left, y_left, abs_tol=tolerance), f"Expecte value at {x_left}={expected_left}, Actual={y_left}. {config}"

                # Check curve goes to expected bound right of center
                x_right = center + 1.5*width
                y_right = po.compute(x_right)
                expected_right = min if flip else max
                assert isclose(expected_left, y_left, abs_tol=tolerance), f"Expecte value at {x_right}={expected_right}, Actual={y_right}. {config}"

                # Check within the curve that we're not yet at the bounds
                x_left = center - 0.5*width
                y_left = po.compute(x_left)
                expected_left = max if flip else min
                assert not isclose(expected_left, y_left, abs_tol=tolerance), f"Expecte value at {x_left}={expected_left}, Actual={y_left}. {config}"

                x_right = center + 0.5*width
                y_right = po.compute(x_right)
                expected_right = min if flip else max
                assert not isclose(expected_left, y_left, abs_tol=tolerance), f"Expecte value at {x_right}={expected_right}, Actual={y_right}. {config}"


class TestRampedValueDataProvider:

    def test_dataprovider(self):
        """ Test the data provider bound to mock pins for its settings
            Non-blocking mode"""

        mock = MockFieldWithRamp("mock")
        mock.block.set_value(0.0)

        # Create the data provider and a ramp_rate input
        data_provider = RampedValueDataProvider(mock.ramp_rate, mock.block, 0.0)
        mock.field.set_data_provider(data_provider)

        #Ramp from 0.0 to "target" at a rate of 10/sec
        mock.ramp_rate.set_value(300.0)
        target=3

        mock.field.set_value(target)

        start = time.time()
        for t in range(0,target+2):
            value = mock.field.get_value()
            elapsed = time.time() - start
            if (t < target):
                assert(t <= value <= t+1)
            else:
                assert(value == target)
            time.sleep(.2)

    def test_dataprovider_with_blocking(self):
        """ Test the data provider bound to mock pins for its settings
            Blocking mode"""

        mock = MockFieldWithRamp("mock")
        mock.block.set_value(1.0)

        # Create the data provider and a ramp_rate input

        data_provider = RampedValueDataProvider(mock.ramp_rate, mock.block, 0.0)
        mock.field.set_data_provider(data_provider)

        mock.ramp_rate.set_value(120.0)
        target=1

        #this should block until ramp is complete
        start = time.time()
        mock.field.set_value(target)

        elapsed = time.time() - start
        value = mock.field.get_value()

        assert(target < (elapsed*2) < target+0.1)
        assert value == target

    def test_dataprovider_create_method_non_blocking(self):
        """ Test the data provider created using the 'create' helper,
            which uses internal pins for the settings data.
            Non-blocking mode"""

        data_provider = RampedValueDataProvider.create(
            ramp_rate_per_min=300, is_blocking=False, starting_value=0.0)

        target = 3
        data_provider.set_value(target)

        for t in range(0,target+2):
            value = data_provider.get_value()
            if (t < target):
                assert(t <= value <= t+1)
            else:
                assert(value == target)
            time.sleep(.2)

    def test_dataprovider_create_method_blocking(self):
        """ Test the data provider created using the 'create' helper,
            which uses internal pins for the settings data.
            Non-blocking mode"""

        data_provider = RampedValueDataProvider.create(
            ramp_rate_per_min=120, is_blocking=True, starting_value=0.0)

        target=1

        #this should block until ramp is complete
        start = time.time()
        data_provider.set_value(target)

        elapsed = time.time() - start
        value = data_provider.get_value()

        assert(target < (elapsed*2) < target+0.1)
        assert value == target


    def test_dataprovider_in_scenario(self):
        """ Test using this data provider in a scenario """
        scenario_file = os.path.join(scenario_root, "ramped_value_data_provider.yaml")

        mock = MockFieldWithRamp("mock")
        mock.ramp_rate.set_value(120.0)
        mock.block.set_value(1.0)

        scenario = SimulationScenario.load_from_yaml(scenario_file)

        # This should set the ramping data provider onto the field pin of the mock
        scenario.run_next_step()

        target = 1.0
        assert(mock.field.get_value() == 0.0)

        start = time.time()
        mock.field.set_value(target)

        elapsed = time.time() - start
        value = mock.field.get_value()

        assert(target <= (elapsed*2) <= target+0.1)
        assert(value == target)


class TestDelayedDataProvider:

    def test_data_provider(self):

        read_delay = 0.1
        write_delay = 0.2
        data_provider = DelayedDataProvider(
            read_delay = read_delay,
            write_delay = write_delay)

        max = 5
        start = time.time()
        expected_delay = 0

        for i in range(0,max):
            data_provider.set_value(i)
            elapsed = time.time() - start
            expected_delay = expected_delay + write_delay
            assert(expected_delay <= elapsed < expected_delay + write_delay)

            value = data_provider.get_value()
            elapsed = time.time() - start
            expected_delay = expected_delay + read_delay
            assert(value == i)
            assert(expected_delay <= elapsed < expected_delay + write_delay)
