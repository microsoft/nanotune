import logging
from functools import partial
from typing import List, Optional, Union, Dict, Tuple, Sequence, Any
import numpy.typing as npt
import numpy as np
import scipy as sc
import itertools
import json
import logging
import copy

from numpy.linalg import inv
from numpy.linalg import multi_dot

from skimage.transform import resize
from scipy.ndimage import gaussian_filter

from qcodes import Instrument, ChannelList, Parameter
from qcodes.dataset.measurements import Measurement
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.dataset.experiment_container import load_by_id

import nanotune as nt
from nanotune.model.node import Node
logger = logging.getLogger(__name__)

LABELS = list(dict(nt.config["core"]["labels"]).keys())
N_2D = nt.config["core"]["standard_shapes"]["2"]
N_1D = nt.config["core"]["standard_shapes"]["1"]
# elem_charge = 1.60217662 * 10e-19
elem_charge = 1
N_lmt_type = Sequence[Tuple[int, int]]


class CapacitanceModel(Instrument):
    """
    Implementation of a general capacitance model an arbitrary number of dots
    and gates. Simulating weakly coupled quantum dots with well localised
    charges, it is a classical description based on two assumptions: (1)
    Coulomb interactions between electrons on dots and in reservoirs are
    parametrised by constant capacitances. (2) The single-particle energy-level
    spectrum is considered independent of electron interactions and the number
    of electrons, meaning that quantum mechanical energy spacings are not taken
    into account.
    The system of electrostatic gates, dots and reservoirs is represented by a
    system of conductors connected via resistors and capacitors

    Being based on
    https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.75.1,
    the implementation uses the same terminology including charge and
    voltage nodes, representing quantum dots and electrostatic gates
    respectively, and electron and hole triple points.

    The capacitor connecting node :math:`j` and node :math:`k` has a
    capacitance :math:`C_{jk}` and stores a charge :math:`q_{jk}`.
    We distinguish between charge and voltage sub-systems, and thus their
    respective sub-matrices:

    .. math::
        :nowrap:

        \mathbf{C} := \\begin{pmatrix}
            \\mathbf{C_{cc}} & \\mathbf{C_{cv}} \\\
            \\mathbf{C_{vc}} & \\mathbf{C_{vv}}
        \\end{pmatrix}

    Diagonal elements of the capacitance matrix, :math:`C_{jj}`, are total
    capacitances of each node and carry the opposite sign of the matrix's
    off-diagonal elements. The off-diagonal elements of
    :math:`\mathbf{\mathbf{C_{cc}}}` are capacitances between charge nodes,
    while the off-diagonal elements of :math:`\mathbf{\mathbf{C_{vv}}}` are
    capacitances between voltage nodes. The elements of
    :math:`\mathbf{\mathbf{C_{cv}}}` are capacitances between voltage and
    charge nodes, and allow to calculate so-called virtual gate coefficients -
    useful knobs in semiconductor qubit experiments.
    """

    def __init__(
        self,
        name: str,
        charge_nodes: Optional[Dict[int, str]] = None,
        voltage_nodes: Optional[Dict[int, str]] = None,
        N: Optional[Sequence[int]] = None,  # charge configuration
        V_v: Optional[Sequence[float]] = None,
        C_cc_off_diags: Optional[Sequence[float]] = None,
        C_cv: Optional[Sequence[Sequence[float]]] = None,
        db_name: str = "capa_model_test.db",
        db_folder: str = nt.config["db_folder"],
    ) -> None:
        """
        Constructor of CapacitanceModel class.

        Args:
            name: Name identifier to be passed to qc.Instrument
            charge_nodes: Dictionary with charge nodes of the model,
                mapping integer node IDs to string labels.
            voltage_nodes: Dictionary with voltage nodes of the model,
                mapping integer node IDs to string labels.
            N: Initial charge configuration, i.e. number of charges on
                each dot. Index of entry corresponds to charge node layout ID.
            V_v: Voltages to set on voltage nodes. Index of entry
                corresponds to charge node layout ID.
            C_cc_off_diags: Capacitances between charge nodes.
            C_cv: Capacitances between charge and voltage nodes.
            db_name: Name of database to store synthetic data.
            db_folder: Path to folder where database is located.
        """
        if charge_nodes is None:
            charge_nodes = {}

        if voltage_nodes is None:
            voltage_nodes = {}

        self.db_name = db_name
        self.db_folder = db_folder
        self._c_l = 0.0
        self._c_r = 0.0
        self._C_cc = np.zeros([len(charge_nodes), len(charge_nodes)]).tolist()

        super().__init__(name)

        c_nodes = ChannelList(self, "charge_nodes", Node, snapshotable=True)
        v_nodes = ChannelList(self, "voltage_nodes", Node, snapshotable=True)

        for nm, vl in charge_nodes.items():
            alias = "chargenode{}".format(nm)
            label = "dot {}".format(nm)
            node = Node(
                parent=self,
                name=alias,
                label=label,
                node_type="charge",
            )
            c_nodes.append(node)
            self.add_submodule(alias, node)

        for nm, vl in voltage_nodes.items():
            alias = "voltagenode{}".format(nm)
            label = vl + " (V[{}])".format(nm)
            node = Node(
                parent=self,
                name=alias,
                label=label,
                node_type="voltage",
            )
            v_nodes.append(node)
            self.add_submodule(alias, node)

        self.add_submodule("charge_nodes", c_nodes)
        self.add_submodule("voltage_nodes", v_nodes)

        if C_cv is None:
            C_cv = np.zeros([len(self.charge_nodes), len(self.voltage_nodes)]).tolist()

        if C_cc_off_diags is None:
            C_cc_off_diags = []
            for off_diag_inx in reversed(range(len(self.charge_nodes) - 1)):
                C_cc_off_diags.append([0.0])  # type: ignore

        if N is None:
            N = [0] * len(self.charge_nodes)
        self.add_parameter(
            "N",
            label="charge configuration",
            unit=None,
            get_cmd=self._get_N,
            set_cmd=self._set_N,
            initial_value=N,
        )

        if V_v is None:
            V_v = [0] * len(self.voltage_nodes)
        self.add_parameter(
            "V_v",
            label="voltage configuration",
            unit=None,
            get_cmd=self._get_V_v,
            set_cmd=self._set_V_v,
            initial_value=V_v,
        )

        self.add_parameter(
            "C_cv",
            label="gate capacitances",
            unit="F",
            get_cmd=self._get_C_cv,
            set_cmd=self._set_C_cv,
            initial_value=C_cv,
        )

        self.add_parameter(
            "C_cc",
            label="dot capacitances",
            unit="F",
            get_cmd=self._get_C_cc,
            set_cmd=self._set_C_cc,
            initial_value=C_cc_off_diags,
        )

        self.add_parameter(
            "c_r",
            label="capacitance to right lead",
            unit="F",
            get_cmd=self._get_c_r,
            set_cmd=self._set_c_r,
            initial_value=0,
        )

        self.add_parameter(
            "c_l",
            label="capacitance to left lead",
            unit="F",
            get_cmd=self._get_c_l,
            set_cmd=self._set_c_l,
            initial_value=0,
        )

    def snapshot_base(
        self,
        update: Optional[bool] = True,
        params_to_skip_update: Optional[Sequence[str]] = None,
    ) -> Dict[Any, Any]:
        """
        Pass on QCoDeS snapshot.
        """

        snap = super().snapshot_base(update, params_to_skip_update)
        return snap

    def set_voltage(
        self,
        n_index: int,
        value: float,
        node_type: str = "voltage",
    ) -> None:
        """Convenience method to set voltages.

        Args:
            n_index: Index of node to set.
            value: Value to set.
            node_type: Which node type to set, either 'voltage' or
                'charge'.
        """
        if node_type == "voltage":
            assert n_index >= 0 and n_index < len(self.voltage_nodes)
            self.voltage_nodes[n_index].v(value)
        elif node_type == "charge":
            assert n_index >= 0 and n_index < len(self.charge_nodes)
            self.charge_nodes[n_index].v(value)
        else:
            logger.error("Unknown node type. Cannot set voltage.")
            raise NotImplementedError

    def set_capacitance(
        self,
        which_matrix: str,
        indices: List[int],
        value: float,
    ) -> None:
        """Convenience function to set capacitances.

        Args:
            which_matrix: String identifier of matrix to set.
                Either 'cv' or 'cc'.
            indices: Indices of capacitances within the matrix.
            value: Capacitance value to set.
        """
        if which_matrix not in ["cv", "cc"]:
            logger.error(
                "Unable to set capacitance. Unknown matrix,"
                + 'choose either "cv" or "cc".'
            )

        if which_matrix == "cc":
            if indices[0] == indices[1]:
                logger.error(
                    "CapacitanceModel: Trying to set diagonals "
                    + "of C_cc matrix which is not possible "
                    + "directly. They are determined by other "
                    + "capacitances of the model."
                )
            C_cc = self.C_cc()
            C_cc[indices[0]][indices[1]] = value
            self.C_cc(C_cc)

        if which_matrix == "cv":
            C_cv = self.C_cv()
            C_cv[indices[0]][indices[1]] = value
            self.C_cv(C_cv)

    def set_Ccv_from_voltage_distance(
        self,
        voltage_node_idx: int,
        dV: float,
        charge_node_idx: int,
    ) -> None:
        """Implements relation between voltage differences and capacitances
        between charge and voltage nodes.

        Args:
            voltage_node_idx: Voltage node index.
            dV: Voltage difference between two charge transitions.
            charge_node_idx: Charge node index.
        """
        capa_val = -elem_charge * dV
        self.set_capacitance(
            "cv",
            [charge_node_idx, voltage_node_idx],
            capa_val,
        )

    def compute_energy(
        self,
        N: Optional[Sequence[int]] = None,
        V_v: Optional[Sequence[float]] = None,
    ) -> float:
        """Compute the total energy of the system.

        Args:
            N: charge configuration, i.e. number of charges on each
                charge node.
            V_v: Voltages to set on voltages nodes.

        Returns:
            float: energy of the system
        """
        if N is None:
            N_self = self.N()
            N_np = np.array(N_self)
        else:
            N_np = np.array(N)

        if V_v is None:
            V_v = np.array(self.V_v())

        C_cc = np.array(self.C_cc())
        C_cv = np.array(self.C_cv())

        U = (elem_charge ** 2) / 2 * multi_dot([N_np, inv(C_cc), N_np])
        U += 1 / 2 * multi_dot([V_v, np.transpose(C_cv), inv(C_cc), C_cv, V_v])
        U += elem_charge * multi_dot([N_np, inv(C_cc), C_cv, V_v])

        return abs(U)

    def determine_N(
        self,
        V_v: Optional[Sequence[float]] = None,
    ) -> List[int]:
        """Determines N by minimizing the total energy of the system.

        Args:
            V_v: Voltages to set on voltages nodes.

        Returns:
            list: Charge state, i.e. number of electrons on each charge node.
        """
        if V_v is None:
            V_v = self.V_v()

        eng_fct = partial(self.compute_energy, V_v=V_v)

        x0 = np.array(self.N()).reshape(-1, 1)
        res = sc.optimize.minimize(
            eng_fct,
            x0,
            method="Nelder-Mead",
            tol=1e-4,
        )
        c_config = np.rint(res.x)

        n_dots = len(c_config)
        energies = []
        c_configs = []

        current_energy = self.compute_energy(N=c_config, V_v=V_v)

        def append_energy(charge_stage: Sequence[int]) -> None:
            charge_stage[charge_stage < 0] = 0  # type: ignore
            energies.append(self.compute_energy(N=charge_stage, V_v=V_v))
            c_configs.append(charge_stage)

        I_mat = np.eye(n_dots)
        # Check if neighbouring ones have lower energy:
        for dot_id in range(n_dots):
            e_hat = I_mat[dot_id]

            append_energy(c_config + e_hat)
            append_energy(c_config - e_hat)

            for other_dot in range(n_dots):
                if other_dot != dot_id:
                    e_hat_other = I_mat[other_dot]
                    append_energy(c_config + e_hat - e_hat_other)

        indx = np.where(np.array(energies) < np.array(current_energy))[0]
        if indx.size > 0:
            min_indx = np.argmin(np.array(energies)[indx.astype(int)])
            min_c_configs = np.array(c_configs)[indx.astype(int)]
            c_config = min_c_configs[min_indx]

        c_config[c_config < 0] = 0
        return c_config.tolist()

    def get_triplepoints(
        self,
        voltage_node_idx: Sequence[int],
        N_limits: N_lmt_type,
    ) -> Tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], List[List[int]]]:
        """Calculates triple points for charge configuration within N_limits

        Args:
            voltage_node_idx: indices of gates to sweep
            N_limit: Min and max values of number of electrons in each dot,
                     defining all charge configurations to consider

        Return:
            np.array: Coordinates of electron triple points
            np.array: Coordinates of hole triple points
            list: List of electron charge configurations
        """

        if len(N_limits) > len(self.N()):
            logger.error(
                "CapacitanceModel.get_triplepoints: "
                + "Infeasible charge configuration supplied."
            )

        c_configs = []
        for n_vals in N_limits:
            c_configs.append(list(range(n_vals[0], n_vals[1] + 1)))

        c_configs = [list(item) for item in itertools.product(*c_configs)]

        coordinates_etp = np.empty([len(c_configs), len(voltage_node_idx)])
        coordinates_htp = np.empty([len(c_configs), len(voltage_node_idx)])

        for ic, c_config in enumerate(c_configs):
            # setting M mostly for monitor purposes
            self.N(list(c_config))
            x_etp, x_htp = self.calculate_triplepoints(
                voltage_node_idx, c_config
            )

            coordinates_etp[ic] = x_etp
            coordinates_htp[ic] = x_htp

        return np.array(coordinates_etp), np.array(coordinates_htp), c_configs

    def calculate_triplepoints(
        self,
        voltage_node_idx: Sequence[int],
        N: Sequence[int],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Determines coordinates in voltage space of triple points
        (electron and hole) for a single charge configuration.

        Args:
            voltage_node_idx: Indices of voltage nodes to be determined
            N: Charge configuration, number of electrons of each charge
                node.

        Return:
            np.array: Coordinates of electron triple points.
            np.array: Coordinates of hole triple points.
        """
        e_tp = partial(
            self.mu_electron_triplepoints,
            voltage_node_idx=voltage_node_idx,
            N=N,
        )
        h_tp = partial(
            self.mu_hole_triplepoints,
            voltage_node_idx=voltage_node_idx,
            N=N,
        )

        x_init = np.array(self.V_v())[voltage_node_idx]

        x_etp = sc.optimize.fsolve(e_tp, x_init)
        x_htp = sc.optimize.fsolve(h_tp, x_init)

        return x_etp, x_htp

    def mu_electron_triplepoints(
        self,
        new_voltages: Sequence[float],
        voltage_node_idx: Sequence[int],
        N: Optional[Sequence[int]] = None,
    ) -> npt.NDArray[np.float64]:
        """Calculates mu_j(N): chemical potentials of all charge nodes for given
        charge configuration N and corresponding to electron triple points.

        Args:
            new_voltages: Voltages to set on voltage nodes.
            voltage_node_idx: Voltage node indices to which the values
                in new_voltages correspond to.
            N: Desired charge configuration, optional. If none supplied
                self.N() is taken.

        Returns:
            np.array: Chemical potentials of electron triple points
                corresponding to electron triple points.
        """
        if N is None:
            N = self.N()

        V_v = self.V_v()
        V_v[voltage_node_idx] = new_voltages

        I_mat = np.eye(len(N))

        out = []
        for dot_id in range(len(N)):
            e_hat = I_mat[dot_id]
            out.append(self.mu(dot_id, N=N + e_hat, V_v=V_v))

        return np.array(out).flatten()

    def mu_hole_triplepoints(
        self,
        new_voltages: Sequence[float],
        voltage_node_idx: Sequence[int],
        N: Optional[Sequence[int]] = None,
    ) -> npt.NDArray[np.float64]:
        """Calculates mu_j(N + e_hat_i): chemical potentials of all charge nodes
            for given charge configuration N and corresponding to hole triple
            points.

        Args:
            new_voltages: values of new gate voltages, to be replaced in
                self.V_v. These are the values scipy.optimize.fsolve is solving
                for
            voltage_node_idx: Voltages nodes indices to which the values
                above correspond to.
            N: Desired charge configuration, if none supplied self.N() is taken.
        """

        if N is None:
            N = self.N()
        V_v = self.V_v()
        V_v[voltage_node_idx] = new_voltages

        I_mat = np.eye(len(N))
        out = []

        for dot_id in range(len(N)):
            e_hat = I_mat[dot_id]
            for other_dot in range(len(N)):
                if other_dot != dot_id:
                    e_hat_other = I_mat[other_dot]
                    out.append(
                        self.mu(dot_id, N=N + e_hat_other + e_hat, V_v=V_v)
                    )

        return np.array(out).flatten()

    def mu(
        self,
        dot_indx: int,
        N: Optional[Sequence[int]] = None,
        V_v: Optional[Sequence[float]] = None,
    ) -> float:
        """Calculates chemical potential of a specific charge node for a given
        charge and voltage configuration.

        Args:
            N: Charge configuration, i.e. the number of electrons on each
                charge node.
            V_v: Voltages to set on (all) voltage nodes.

        Returns:
            float: Chemical potential.
        """

        if N is None:
            N_np = np.array(self.N())
        else:
            N_np = np.array(N)

        if V_v is None:
            V_v_np = np.array(self.V_v())
        else:
            V_v_np = np.array(V_v)

        C_cc = np.array(self.C_cc())
        C_cv = np.array(self.C_cv())

        e_hat = np.zeros(len(N_np)).astype(int)
        e_hat[dot_indx] = 1

        pot = -(elem_charge ** 2) / 2 * multi_dot([e_hat, inv(C_cc), e_hat])
        pot += (elem_charge ** 2) * multi_dot([N_np, inv(C_cc), e_hat])
        pot += elem_charge * multi_dot([e_hat, inv(C_cc), C_cv, V_v_np])

        return pot

    def sweep_voltages(
        self,
        voltage_node_idx: Sequence[int],  # the one we want to sweep
        voltage_ranges: Sequence[Tuple[float, float]],
        n_steps: Sequence[int] = N_2D,
        line_intensity: float = 1.0,
        broaden: bool = True,
        add_noise: bool = True,
        kernel_widths: Tuple[float, float] = (1.0, 1.0),
        target_snr_db: float = 100,
        normalize: bool = True,
        known_regime: str = 'doubledot',
        known_quality: Optional[int] = None,
        add_charge_jumps: bool = False,
        jump_freq: float = 0.001,
    ) -> Optional[int]:
        """Sweep two voltage nodes to measure a charge diagram. Determines
        charge transitions by computing the number of degeneracies using
        get_number_of_degeneracies. Applies a Gaussien filter to broaden
        transitions and adds random normal noise and optionally charge jumps.
        The diagram is saved into .db using QCoDeS.

        Args:
            voltage_node_idx: Voltage node indices to sweep.
            voltage_ranges: Voltage ranges to sweep.
            n_steps: Number of steps of the measurement.
            line_intensity: Multiplication factor of number of
                degeneracies, resulting in the desired peak hight before
                normalization.
            kernel_widths: Sigmas of Gaussian filter used for broadening
                of charge transitions.
            target_snr_db: Target signal-to-noise ratio used to
                calculate amplitude of random normal noise.
            e_temp: Electron temperature, used as absolute tolerance in
                energy at which two charge states count as degenerate.
            known_regime: Label to be saved in metadata.
            known_quality: Quality to be saved in metadata.
            add_charge_jumps: Whether or not to add random charge jumps.
            jump_freq: Average frequency at which optional charge jumps
                should occur.

        Returns:
            int: QCoDeS data run ID
        """
        self.set_voltage(voltage_node_idx[0], np.min(voltage_ranges[0]))
        self.set_voltage(voltage_node_idx[1], np.min(voltage_ranges[1]))

        voltage_x = np.linspace(
            np.min(voltage_ranges[0]), np.max(voltage_ranges[0]), n_steps[0]
        )

        voltage_y = np.linspace(
            np.min(voltage_ranges[1]), np.max(voltage_ranges[1]), n_steps[1]
        )
        signal = np.zeros(n_steps)

        if add_charge_jumps:
            additional_charges = np.ones(n_steps)
            s = 1

            poisson = np.random.poisson(lam=jump_freq, size=n_steps)
            poisson[poisson > 1] = 1
            for ix in range(n_steps[0]):
                for iy in range(n_steps[1]):
                    if poisson[ix, iy] == 1:
                        s *= -1
                    additional_charges[ix, iy] *= s
            additional_charges = np.array(additional_charges)
            additional_charges = (additional_charges + 1) / 2
        else:
            additional_charges = np.zeros(n_steps)

        for ivx, x_val in enumerate(voltage_x):
            self.set_voltage(voltage_node_idx[1], voltage_y[0])
            self.set_voltage(voltage_node_idx[0], x_val)

            for ivy, y_val in enumerate(voltage_y):
                self.set_voltage(voltage_node_idx[1], y_val)

                N_curr = self.determine_N()
                n_idx = int(np.random.randint(len(N_curr), size=1))
                N_curr[n_idx] += additional_charges[ivx, ivy]
                self.N(N_curr)

                dU = self.get_energy_differences_to_adjacent_charge_states(
                    N_current=N_curr,
                )
                signal[ivx, ivy] = np.sum(np.reciprocal(dU)) * line_intensity

        signal = (signal - np.mean(signal))

        if broaden:
            signal = self._make_it_real(signal, kernel_widths=kernel_widths)
        if add_noise:
            signal = self._add_noise(signal, target_snr_db=target_snr_db)

        if normalize:
            signal = signal / np.max(signal)

        if known_quality is None:
            if target_snr_db > 2:
                quality = 1
            else:
                quality = 0
        else:
            quality = known_quality

        dataid = self._save_to_db(
            [self.voltage_nodes[voltage_node_idx[0]].v,
             self.voltage_nodes[voltage_node_idx[1]].v],
            [voltage_x.tolist(), voltage_y.tolist()],
            signal,
            nt_label=[known_regime],
            quality=quality,
        )

        return dataid

    def sweep_voltage(
        self,
        voltage_node_idx: int,  # the one we want to sweep
        voltage_range: Sequence[float],
        n_steps: int = N_1D[0],
        line_intensity: float = 1.0,
        kernel_width: Sequence[float] = [1.0],
        target_snr_db: float = 100.0,
        normalize: bool = True,
    ) -> Optional[int]:
        """Sweep one voltage to measure Coulomb oscillations. Determines
        charge transitions by computing the number of degeneracies using
        get_number_of_degeneracies. Applies a Gaussien filter to broaden
        transitions and adds random normal noise and optionally charge jumps.
        The diagram is saved into .db using QCoDeS.

        Args:
            voltage_node_idx: Voltage node indices to sweep.
            voltage_ranges: Voltage ranges to sweep.
            n_steps: Number of steps of the measurement.
            line_intensity: Multiplication factor of number of
                degeneracies, resulting in the desired peak hight before
                normalization.
            e_temp: Electron temperature, used as absolute tolerance in
                energy at which two charge states count as degenerate.
            known_quality: Quality to be saved in metadata.
            add_charge_jumps: Whether or not to add random charge jumps.
            jump_freq: Average frequency at which optional charge jumps
                should occur.

        Returns:
            int: QCoDeS data run ID.
        """
        self.set_voltage(voltage_node_idx, np.min(voltage_range))
        signal = np.zeros(n_steps)

        voltage_x = np.linspace(
            np.min(voltage_range), np.max(voltage_range), n_steps
        )
        for iv, v_val in enumerate(voltage_x):
            self.set_voltage(voltage_node_idx, v_val)
            N_current = self.determine_N()
            self.N(N_current)

            dU = self.get_energy_differences_to_adjacent_charge_states(
                N_current=N_current,
            )
            signal[iv] = np.sum(np.reciprocal(dU)) * line_intensity

        signal = (signal - np.mean(signal))
        signal = self._make_it_real(signal, kernel_widths=kernel_width)
        signal = self._add_noise(signal, target_snr_db=target_snr_db)

        if normalize:
            signal = signal / np.max(signal)

        dataid = self._save_to_db(
            [self.voltage_nodes[voltage_node_idx].v],
            [voltage_x.tolist()],
            signal,
            nt_label=["coulomboscillation"],
        )

        return dataid

    def sweep_bias_and_voltage(
        self,
        voltage_node_idx: int,  # the one we want to sweep
        voltage_range: Sequence[float],
        bias_range: Tuple[float, float],
        n_steps: Sequence[int] = N_2D,
        line_intensity: float = 1.0,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ """

        self.set_voltage(voltage_node_idx, np.min(voltage_range))

        voltage_x = np.linspace(
            np.min(voltage_range), np.max(voltage_range), n_steps[0]
        )

        bias_steps = np.linspace(
            np.min(bias_range), np.max(bias_range), n_steps[1]
        )
        signal = np.zeros(n_steps)
        co_tunn = np.zeros(n_steps)

        for iv, v_val in enumerate(voltage_x):
            self.set_voltage(voltage_node_idx, v_val)
            N_current = self.determine_N()
            self.N(N_current)

            dU = self.get_energy_differences_to_excited_charge_states(
                N_current=N_current,
                how_many=1,
            )
            for ib, b_val in enumerate(bias_steps):
                if (dU <= abs(b_val)).any():
                    signal[iv, ib] = line_intensity
                    engs_in_bias = dU[dU <= abs(b_val)]
        #             signal[iv, ib] = np.sum(np.reciprocal(engs_in_bias)) * line_intensity
                    signal[iv, ib] = len(engs_in_bias) * line_intensity
                else:
                    signal[iv, ib] = 0

                    N_plus_one = copy.deepcopy(N_current)
                    N_plus_one[0] = N_plus_one[0] + 1.  # type: ignore
                    add_rate = self.get_co_tunneling_rate(
                        b_val,
                        self.mu(0, N=N_current),
                        self.mu(0, N=N_plus_one),
                    )
                    co_tunn[iv, ib] += add_rate

        signal = (signal - np.mean(signal))/np.std(signal)
        co_tunn = (co_tunn - np.mean(co_tunn))/np.std(co_tunn)

        return signal, co_tunn

    def get_co_tunneling_rate(
        self,
        bias: float,
        mu_n: float,
        mu_n_plus_one: float,
        source_rate_N: float = 0.3,
        source_rate_N_plus_one: float = 0.2,
        drain_rate_N: float = 0.3,
        drain_rate_N_plus_one: float = 0.2,
        h_bar: float = 1,
    ) -> float:

        mu_source = bias
        mu_drain = 0

        first_term = (source_rate_N * drain_rate_N)
        first_term /= ((mu_drain - mu_n)*(mu_source - mu_n))
        second_term = (source_rate_N_plus_one * drain_rate_N_plus_one)
        second_term /= ((mu_drain - mu_n_plus_one)*(mu_source - mu_n_plus_one))
        third_term = np.log((mu_n_plus_one - mu_drain)/(mu_drain - mu_n))
        third_term -= np.log((mu_n_plus_one - mu_source)/(mu_source - mu_n))
        third_term /= ((mu_n_plus_one - mu_n) * (mu_source - mu_drain))

        co_tunneling_rate = 2 * np.pi/h_bar * (first_term + second_term) * bias
        co_tunneling_rate += third_term

        return co_tunneling_rate

    def get_energy_differences_to_adjacent_charge_states(
        self,
        N_current: Optional[Sequence[int]] = None,
        V_v: Optional[Sequence[float]] = None,
    ) -> int:
        """
        """

        n_dots = len(self.charge_nodes)

        if N_current is None:
            N_current = self.N()
        else:
            self.N(N_current)

        if V_v is None:
            V_v = self.V_v()
        else:
            self.V_v(V_v)

        I_mat = np.eye(n_dots)
        energies = []

        for dot_id in range(n_dots):
            e_hat = I_mat[dot_id]

            charge_state = N_current + e_hat
            charge_state[charge_state < 0] = 0
            energies.append(self.compute_energy(N=charge_state, V_v=V_v))

            charge_state = N_current - e_hat
            charge_state[charge_state < 0] = 0
            energies.append(self.compute_energy(N=charge_state, V_v=V_v))

            for other_dot in range(n_dots):
                if other_dot != dot_id:
                    e_hat_other = I_mat[other_dot]
                    charge_state = N_current + e_hat - e_hat_other
                    charge_state[charge_state < 0] = 0
                    energies.append(
                        self.compute_energy(
                            N=charge_state, V_v=V_v,
                        )
                    )

        current_energy = self.compute_energy(N=N_current, V_v=V_v)
        dU = abs(np.array(energies) - current_energy)

        return dU[dU != 0]

    def get_energy_differences_to_excited_charge_states(
        self,
        N_current: Optional[Sequence[int]] = None,
        V_v: Optional[Sequence[float]] = None,
        how_many: int = 3,
    ) -> List[float]:
        """
        """

        n_dots = len(self.charge_nodes)

        if N_current is None:
            N_current = self.N()
        else:
            self.N(N_current)

        if V_v is None:
            V_v = self.V_v()
        else:
            self.V_v(V_v)

        I_mat = np.eye(n_dots)
        energies = []

        for dot_id in range(n_dots):
            e_hat = I_mat[dot_id]

            for add_e in range(how_many):
                charge_state = N_current + (add_e+1)*e_hat
                charge_state[charge_state < 0] = 0
                energies.append(self.compute_energy(N=charge_state, V_v=V_v))

            charge_state = N_current - e_hat
            charge_state[charge_state < 0] = 0
            energies.append(self.compute_energy(N=charge_state, V_v=V_v))

            for other_dot in range(n_dots):
                if other_dot != dot_id:
                    e_hat_other = I_mat[other_dot]
                    for add_e in range(how_many):
                        charge_state = N_current + (add_e+1)*e_hat - e_hat_other
                        charge_state[charge_state < 0] = 0
                        energies.append(
                            self.compute_energy(
                                N=charge_state, V_v=V_v,
                            )
                        )

        current_energy = self.compute_energy(N=N_current, V_v=V_v)
        dU = abs(np.array(energies) - current_energy)

        return dU[dU != 0].tolist()

    def _make_it_real(
        self,
        diagram: npt.NDArray[np.float64],
        kernel_widths: Union[float, Sequence[float]] = [3.0, 3.0],
    ) -> npt.NDArray[np.float64]:
        """Uses a Gaussian filter to broaden a stickfigure diagram.

        Args:
            diagram: The previously computed stickfigure diagram
            kernel_width (list, float): Width of the Gaussian kernel. Can be a
                single number or list of the same length as the diagram's
                dimensions.

        Return:
            np.ndarray: Gaussian blurred diagram.
        """
        org_shape = diagram.shape
        diagram = gaussian_filter(
            diagram, sigma=kernel_widths, mode="constant", truncate=1
        )

        return resize(diagram, org_shape)

    def _add_noise(
        self,
        diagram: npt.NDArray[np.float64],
        target_snr_db: float = 10.0,
    ) -> npt.NDArray[np.float64]:
        """Adds normally distributed random noise to a diagram to match the
        desired signal to noise ratio.

        Args:
            diagram: Noise free diagram
            target_snr_db: Target signal to noise ratio in dB

        Return:
            np.ndarray: Diagram with normally distributed random noise added.
        """

        d_shape = diagram.shape
        sig_mean_power = np.mean(diagram ** 2 / 2)
        sig_avg_db = 10 * np.log10(sig_mean_power)

        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_power = 10 ** (noise_avg_db / 10)

        mean_noise = 0
        noise = np.random.normal(mean_noise, noise_avg_power, np.prod(d_shape))
        noise = noise.reshape(*d_shape)

        return diagram + noise

    def _save_to_db(
        self,
        parameters: Sequence[Parameter],
        setpoints: Sequence[Sequence[float]],
        data: npt.NDArray[np.float64],
        nt_label: Sequence[str],
        quality: int = 1,
    ) -> Union[None, int]:
        """Save data to a database using QCoDeS.

        Args:
            parameters: List of QCoDeS parameters to register for
                measurement.
            setpoints: List of setpoints, i.e. voltage values.
            data: The measurement to save.
            nt_label: List of machine learning/nanotune labels to save.
            quality: The measurement's quality, to be saved as metadata.

        Returns:
            int: QCoDeS data run id.
        """
        # TODO: Save the data as arrays and remove for loops. To be updated to
        # new QCoDeS data formats

        dummy_lockin = DummyInstrument("dummy_lockin", gates=["R"])
        if len(parameters) not in [1, 2]:
            logger.error("Only 1D and 2D sweeps supported right now.")
            return None

        meas = Measurement()

        if len(parameters) == 1:
            meas.register_parameter(parameters[0])
            meas.register_parameter(dummy_lockin.R, setpoints=(parameters[0],))

            with meas.run() as datasaver:
                for x_indx, x_val in enumerate(setpoints[0]):
                    parameters[0](x_val)
                    datasaver.add_result(
                        (parameters[0], x_val), (dummy_lockin.R, data[x_indx])
                    )

                dataid = datasaver.run_id

        if len(parameters) == 2:
            meas.register_parameter(parameters[0])
            meas.register_parameter(parameters[1])
            meas.register_parameter(
                dummy_lockin.R, setpoints=(parameters[0], parameters[1])
            )

            with meas.run() as datasaver:
                for x_indx, x_val in enumerate(setpoints[0]):
                    parameters[0](x_val)
                    for y_indx, y_val in enumerate(setpoints[1]):
                        parameters[1](y_val)
                        datasaver.add_result(
                            (parameters[0], x_val),
                            (parameters[1], y_val),
                            (dummy_lockin.R, data[x_indx, y_indx]),
                        )

                dataid = datasaver.run_id

        ds = load_by_id(dataid)

        meta_add_on: Dict[str, Any] = dict.fromkeys(
            nt.config["core"]["meta_fields"], None
        )
        meta_add_on["device_name"] = self.name
        nm = dict.fromkeys(["transport", "rf"], (0, 1))
        meta_add_on["normalization_constants"] = nm

        ds.add_metadata(nt.meta_tag, json.dumps(meta_add_on))

        current_label = dict.fromkeys(LABELS, 0)
        for label in nt_label:
            if label is not None:
                if label not in LABELS:
                    logger.error(f"CapacitanceModel: Invalid label: {label}")
                    raise ValueError
                current_label[label] = 1
                current_label["good"] = quality

        for label, value in current_label.items():
            ds.add_metadata(label, value)

        dummy_lockin.close()

        return dataid

    def determine_sweep_voltages(
        self,
        voltage_node_idx: Sequence[int],
        V_v: Optional[Sequence[float]] = None,
        N_limits: Optional[N_lmt_type] = None,
    ) -> List[List[float]]:
        """Determines voltages to sweep to measure specific charge transitions.

        Args:
            voltage_node_idx: Indices of voltages nodes to sweep.
            V_v: Voltage configuration of all gates.
            N_limits: Charge configuration ranges to measure. E.g. for a double
                dot to sweep over empty dots to both having 3 electrons: [(0, 3), (0, 3)]

        Returns:
            list: Nested list of voltages limits to sweep. Example double dot:
                [[gate1_min_voltage, gate1_max_voltage], [gate2_min_voltage, gate2_max_voltage]]
        """
        if V_v is None:
            V_v = self.V_v()
        if N_limits is None:
            N_limits = [(0, 1)] * len(self.N())

        N_init = []
        for did in range(len(N_limits)):
            N_init.append(N_limits[did][0])

        def eng_sub_fct(N, swept_voltages):
            curr_V = V_v
            for iv, v_to_sweep in enumerate(voltage_node_idx):
                curr_V[v_to_sweep] = swept_voltages[iv]
            return self.compute_energy(N=N, V_v=curr_V)

        eng_fct = partial(eng_sub_fct, N_init)
        x0 = []
        for v_to_sweep in voltage_node_idx:
            x0.append(V_v[v_to_sweep])
        res = sc.optimize.minimize(
            eng_fct,
            x0,
            method="Nelder-Mead",
            tol=1e-4,
        )
        V_init_config = res.x

        N_stop = []
        for did in range(len(N_limits)):
            N_stop.append(N_limits[did][1])

        eng_fct = partial(eng_sub_fct, N_stop)

        x0 = V_init_config

        res = sc.optimize.minimize(
            eng_fct,
            x0,
            method="Nelder-Mead",
            tol=1e-4,
        )
        V_stop_config = res.x

        return list(zip(V_init_config, V_stop_config))  # type: ignore

    def _get_N(self) -> List[int]:
        """ QCoDeS parameter getter for charge configuration N. """
        for inn, c_n in enumerate(self.charge_nodes):
            self._N[inn] = c_n.n()
        return self._N

    def _set_N(self, value: List[int]):
        """ QCoDeS parameter setter for charge configuration N. """
        value = list(value)
        self._N = value
        for iv, val in enumerate(value):
            self.charge_nodes[iv].n(int(val))

    def _get_V_v(self) -> List[float]:
        """ QCoDeS parameter getter for voltage configuration V_v. """
        for inn, v_n in enumerate(self.voltage_nodes):
            self._V_v[inn] = v_n.v()
        return self._V_v

    def _set_V_v(self, value: List[float]):
        """ QCoDeS parameter setter for voltage configuration V_v. """
        value = list(value)
        self._V_v = value
        for iv, val in enumerate(value):
            self.voltage_nodes[iv].v(val)

    def _get_C_cc(self) -> List[List[float]]:
        """ QCoDeS parameter getter for dot capacitance matrix C_cc. Diagonals
        of C_cc is the sum of all capacitances connected to the respective
        charge node, calculated in _get_C_cc_diagonals.
        """
        current_C_cc = np.array(self._C_cc)
        diagonals = self._get_C_cc_diagonals()
        for dot_ind in range(len(self.charge_nodes)):
            current_C_cc[dot_ind, dot_ind] = diagonals[dot_ind, dot_ind]

        self._C_cc = current_C_cc.tolist()
        return self._C_cc

    def _set_C_cc(self, off_diagonals: List[List[float]]):
        """ QCoDeS parameter setter for dot capacitance matrix C_cc. """
        self._C_cc = np.zeros([len(self.charge_nodes), len(self.charge_nodes)])
        for dinx, diagonal in enumerate(off_diagonals):
            if len(diagonal) != (len(self.charge_nodes) - dinx - 1):
                logger.error(
                    "CapacitanceModel: Unable to set C_cc. "
                    + "Please specify off diagonals in a list of "
                    + "lists: [[1st off diagonal], "
                    + "[2nd off diagonal]]"
                )
            self._C_cc += np.diag(diagonal, k=dinx + 1)
            self._C_cc += np.diag(diagonal, k=-dinx - 1)

        self._C_cc += self._get_C_cc_diagonals()
        self._C_cc = self._C_cc.tolist()

    def _get_C_cc_diagonals(self) -> npt.NDArray[np.float64]:
        """Getter for diagonal values of dot capacitance matrix C_cc.
        We assume that every dot is coupled to every other, meaning that
        that if three or more dots are aligned the first will have a capacitive
        coupling to the last. In the same manner, all dots are coupled to the
        leads. Change if necessary.
        """
        C_cv_sums = np.sum(np.absolute(np.array(self._C_cv)), axis=1)
        # from other dots:
        off_diag = self._C_cc - np.diag(np.diag(self._C_cc))
        off_diag_sums = np.sum(np.absolute(off_diag), axis=1)

        diag = C_cv_sums + off_diag_sums
        diag += np.absolute(self._c_r) + np.absolute(self._c_r)

        return np.diag(diag)

    def _get_C_cv(self) -> List[List[float]]:
        """ QCoDeS parameter getter for dot capacitance matrix C_cv. """
        return self._C_cv

    def _set_C_cv(self, value: List[List[float]]):
        """ QCoDeS parameter setter for dot capacitance matrix C_cv. Updates dot
        capacitance matrix C_cc as its diagonals depend on C_cv
        """
        self._C_cv = value
        _ = self._get_C_cc()

    def _get_c_r(self) -> float:
        """ QCoDeS parameter getter for lead capacitance R(right). """
        return self._c_r

    def _set_c_r(self, value: float):
        """ QCoDeS parameter setter for lead capacitance R(right). Updates dot
        capacitance matrix C_cc as its diagonals depend on c_r
        """
        self._c_r = value
        try:
            _ = self._get_C_cc()
        except Exception:
            logger.warning(
                "Setting CapacitanceModel.c_r: Unable to update C_cc"
            )
            pass

    def _get_c_l(self) -> float:
        """ QCoDeS parameter getter for lead capacitance L(eft). """
        return self._c_l

    def _set_c_l(self, value: float):
        """ QCoDeS parameter setter for lead capacitance L(eft). Updates dot
        capacitance matrix C_cc as its diagonals depend on c_l
        """
        self._c_l = value
        try:
            _ = self._get_C_cc()
        except Exception:
            logger.warning(
                "Setting CapacitanceModel.c_l: Unable to update C_cc"
            )
            pass
