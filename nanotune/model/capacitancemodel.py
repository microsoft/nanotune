import logging
from functools import partial
from typing import List, Optional, Union, Dict, Tuple, Sequence, Any, Sequence
import numpy as np
import scipy as sc
import itertools
import json

import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy.linalg import multi_dot
from operator import itemgetter
from tabulate import tabulate

from skimage.transform import resize
from scipy.ndimage import gaussian_filter

import qcodes as qc
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

elem_charge = 1.60217662 * 10e-19
# elem_charge = 1.60217662
# elem_charge = 1
N_lmt_type = Sequence[Tuple[int, int]]

# TODO: make variable names consistent. E.g: N/c_config/charge_configuration
# TODO: check C_cc_off_diags. Initialised correctly?


class CapacitanceModel(Instrument):
    """
    Based on https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.75.1

    As in this paper, we call the two types of triple point 'electron' and
    'hole' triple points.
    """

    def __init__(
        self,
        name: str,
        charge_nodes: Optional[Dict[int, str]] = None,
        voltage_nodes: Optional[Dict[int, str]] = None,
        dot_handles: Optional[Dict[int, str]] = None,
        N: Optional[Sequence[int]] = None,  # charge configuration
        V_v: Optional[Sequence[float]] = None,
        C_cc_off_diags: Optional[Sequence[float]] = None,
        C_cv: Optional[Sequence[Sequence[float]]] = None,
        db_name: str = "capa_model_test.db",
        db_folder: str = nt.config["db_folder"],
    ) -> None:

        self.db_name = db_name
        self.db_folder = db_folder
        self._C_l = 0.0
        self._C_r = 0.0
        self._C_cc = np.zeros([len(charge_nodes), len(charge_nodes)])

        if charge_nodes is None:
            charge_nodes = {}

        if voltage_nodes is None:
            voltage_nodes = {}

        if dot_handles is None:
            dot_handles = {}

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
            C_cv = np.zeros([len(self.charge_nodes), len(self.voltage_nodes)])

        if C_cc_off_diags is None:
            C_cc_off_diags = []
            for off_diag_inx in reversed(range(len(self.charge_nodes) - 1)):
                C_cc_off_diags.append([0.0])

        self.add_parameter(
            "dot_handles",
            label="main dot handles",
            unit=None,
            get_cmd=self._get_dot_handles,
            set_cmd=self._set_dot_handles,
            initial_value=dot_handles,
        )

        self.add_parameter(
            "charge_node_mapping",
            label="charge node name mapping",
            unit=None,
            get_cmd=self._get_charge_node_mapping,
            set_cmd=self._set_charge_node_mapping,
            initial_value=charge_nodes,
        )

        if N is None:
            N = np.zeros(len(self.charge_nodes))

        self.add_parameter(
            "N",
            label="charge configuration",
            unit=None,
            get_cmd=self._get_N,
            set_cmd=self._set_N,
            initial_value=N,
        )

        if V_v is None:
            V_v = np.zeros(len(self.voltage_nodes))
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
            "C_R",
            label="capacitance to right lead",
            unit="F",
            get_cmd=self._get_C_R,
            set_cmd=self._set_C_R,
            initial_value=0,
        )

        self.add_parameter(
            "C_L",
            label="capacitance to left lead",
            unit="F",
            get_cmd=self._get_C_L,
            set_cmd=self._set_C_L,
            initial_value=0,
        )

    def _get_dot_handles(self) -> Dict[int, str]:
        return self._dot_handles

    def _set_dot_handles(self, value: Dict[int, str]):
        self._dot_handles = value

    def _get_charge_node_mapping(self) -> Dict[int, str]:
        return self._charge_node_mapping

    def _set_charge_node_mapping(self, value: Dict[int, str]):
        self._charge_node_mapping = value

    def _get_N(self) -> List[int]:
        for inn, c_n in enumerate(self.charge_nodes):
            self._N[inn] = c_n.n()
        return self._N

    def _set_N(self, value: List[int]):
        value = list(value)
        self._N = value
        for iv, val in enumerate(value):
            self.charge_nodes[iv].n(int(val))

    def _get_V_v(self) -> List[float]:
        for inn, v_n in enumerate(self.voltage_nodes):
            self._V_v[inn] = v_n.v()
        return self._V_v

    def _set_V_v(self, value: List[float]):
        value = list(value)
        self._V_v = value
        for iv, val in enumerate(value):
            self.voltage_nodes[iv].v(val)

    def _get_C_cc(self) -> List[List[float]]:
        # Get diagonals: sum of all capacitances attached to it.
        current_C_cc = np.array(self._C_cc)
        diagonals = self._get_C_cc_diagonals()
        for dot_ind in range(len(self.charge_nodes)):
            current_C_cc[dot_ind, dot_ind] = diagonals[dot_ind, dot_ind]

        self._C_cc = current_C_cc.tolist()
        return self._C_cc

    def _set_C_cc(self, off_diagonals: List[List[float]]):

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

    def _get_C_cc_diagonals(self) -> np.ndarray:
        """
        Here we assume that every dot is coupled to every other. This means
        that if three or more dots are aligned the first will have a capacitive
        coupling to the last. In the same manner, all dots are coupled to the
        leads. Change if necessary.
        """
        C_cc = self._C_cc
        C_cv_sums = np.sum(np.absolute(np.array(self._C_cv)), axis=1)
        # from other dots:
        off_diag = self._C_cc - np.diag(np.diag(self._C_cc))
        off_diag_sums = np.sum(np.absolute(off_diag), axis=1)

        diag = C_cv_sums + off_diag_sums
        diag += np.absolute(self._C_r) + np.absolute(self._C_r)

        return np.diag(diag)

    def _get_C_cv(self) -> List[List[float]]:
        return self._C_cv

    def _set_C_cv(self, value: List[List[float]]):
        self._C_cv = value
        # update values in C_cc:
        _ = self._get_C_cc()

    def _get_C_R(self) -> float:
        return self._C_r

    def _set_C_R(self, value: float):
        self._C_r = value
        try:
            _ = self._get_C_cc()
        except Exception:
            logger.warning(
                "Setting CapacitanceModel.C_R: Unable to update C_cc"
            )
            pass

    def _get_C_L(self) -> float:
        return self._C_l

    def _set_C_L(self, value: float):
        self._C_l = value
        try:
            _ = self._get_C_cc()
        except Exception:
            logger.warning(
                "Setting CapacitanceModel.C_L: Unable to update C_cc"
            )
            pass

    def snapshot_base(
        self,
        update: Optional[bool] = True,
        params_to_skip_update: Optional[Sequence[str]] = None,
    ) -> Dict[Any, Any]:

        snap = super().snapshot_base(update, params_to_skip_update)

        return snap

    def set_voltage(
        self,
        n_index: int,
        value: float,
        n_type: str = "voltage",
    ) -> None:
        """ Convenience method to set voltages. """
        if n_type == "voltage":
            assert n_index >= 0 and n_index < len(self.voltage_nodes)
            self.voltage_nodes[n_index].v(value)
        elif n_type == "charge":
            assert n_index >= 0 and n_index < len(self.charge_nodes)
            self.charge_nodes[n_index].v(value)
        else:
            logger.error("Unknown node type. Can not set voltage.")
            raise NotImplementedError

    def set_capacitance(
        self, which_matrix: str, indexes: List[int], value: float
    ) -> None:
        """ Convenience function to set capacitances. """
        if which_matrix not in ["cv", "cc"]:
            logger.error(
                "Unable to set capacitance. Unknown matrix,"
                + 'choose either "cv" or "cc".'
            )

        if which_matrix == "cc":
            if indexes[0] == indexes[1]:
                logger.error(
                    "CapacitanceModel: Trying to set diagonals "
                    + "of C_cc matrix which is not possible "
                    + "directly. They are determined by other "
                    + "capacitances of the model."
                )
            C_cc = self.C_cc()
            C_cc[indexes[0]][indexes[1]] = value
            self.C_cc(C_cc)

        if which_matrix == "cv":
            C_cv = self.C_cv()
            C_cv[indexes[0]][indexes[1]] = value
            self.C_cv(C_cv)

    def set_Ccv_from_voltage_distance(
        self,
        v_node_idx: int,
        dV: float,
        dot_indx: int,
    ) -> None:
        """
        Formulas relating voltage differences to capacitance
        """
        capa_val = -elem_charge * dV
        self.set_capacitance("cv", [dot_indx, v_node_idx], capa_val)

    def compute_energy(
        self,
        N: Optional[Sequence[int]] = None,
        V_v: Optional[Sequence[float]] = None,
    ) -> float:
        """ Compute the total energy of dots """
        if N is None:
            N_self = self.N()
            N = np.array(N_self).reshape(len(N_self), 1)
        if V_v is None:
            V_v = self.V_v()

        C_cc = np.array(self.C_cc())
        C_cv = np.array(self.C_cv())

        U = (elem_charge ** 2) / 2 * multi_dot([N, inv(C_cc), N])
        U += 1 / 2 * multi_dot([V_v, np.transpose(C_cv), inv(C_cc), C_cv, V_v])
        U += elem_charge * multi_dot([N, inv(C_cc), C_cv, V_v])

        return abs(U)

    def determine_N(
        self,
        V_v: Optional[Sequence[float]] = None,
    ) -> List[int]:
        """
        Determine N by minimizing the energy
        """
        if V_v is None:
            V_v = self.V_v()

        eng_fct = partial(self.compute_energy, V_v=V_v)

        x0 = np.array(self.N()).reshape(-1, 1)
        res = sc.optimize.minimize(
            eng_fct,
            x0,
            #    method='COBYLA',
            #    method='Powell',
            method="Nelder-Mead",
            tol=1e-4,
        )
        c_config = np.rint(res.x)

        n_dots = len(c_config)
        energies = []
        c_configs = []

        current_energy = self.compute_energy(N=c_config, V_v=V_v)

        def append_energy(charge_stage: Sequence[int]) -> None:
            if not np.any(np.array(charge_stage) < 0):
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

        return c_config.tolist()

    def get_triplepoints(
        self,
        v_node_idx: Sequence[int],
        N_limits: N_lmt_type,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """Calculate triple points for charge configuration within N_limits

        Args:
            v_node_idx: Indexes of gates to sweep
            N_limit: Min and max values of number of electrons in each dot,
                     defining all charge configurations to consider

        Return:
            List of electron charge configurations (first array) and hole
            triple points (second array).

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

        coordinates_etp = np.empty([len(c_configs), len(v_node_idx)])
        coordinates_htp = np.empty([len(c_configs), len(v_node_idx)])

        for ic, c_config in enumerate(c_configs):
            # setting M mostly for monitor purposes
            self.N(list(c_config))
            x_etp, x_htp = self.calculate_triplepoint(
                v_node_idx, np.array(c_config)
            )

            coordinates_etp[ic] = x_etp
            coordinates_htp[ic] = x_htp

        return np.array(coordinates_etp), np.array(coordinates_htp), c_configs

    def calculate_triplepoint(
        self,
        v_node_idx: Sequence[int],
        N: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine coordinates in voltage space of triple points
        (electron and hole) for one charge configuration.

        Args:
            v_node_idx: Indexes of voltage nodes to be determined
            N: Charge configuration, number of electrons in each dot.

        Return:
            Coordinates of electron (first array) and hole (second array)
            triple points.
        """
        e_tp = partial(self.triplepoint_e, v_node_idx=v_node_idx, N=N)
        h_tp = partial(self.triplepoint_h, v_node_idx=v_node_idx, N=N)

        x_init = np.array(self.V_v())[v_node_idx]

        x_etp = sc.optimize.fsolve(e_tp, x_init)
        x_htp = sc.optimize.fsolve(h_tp, x_init)

        return x_etp, x_htp

    def triplepoint_e(
        self,
        new_voltages: Sequence[float],
        v_node_idx: Sequence[int],
        N: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Calculate chemical potentials of all dots for charge configuration
            N, corresponding to electron triple points. mu_j(N)
        Args:
            new_voltages: values of new gate voltages, to be replaced in
                          self.V_v. These are the values scipy.optimize.fsolve
                          is solving for
            v_node_idx: Voltages nodes indexes to which the values above
                        correspond to.
            N: Desired charge configuration, if none supplied self.N is taken.
        """
        if N is None:
            N = self.N()

        V_v = np.array(self.V_v())
        V_v[v_node_idx] = new_voltages

        I_mat = np.eye(len(N))

        out = []
        for dot_id in range(len(N)):
            e_hat = I_mat[dot_id]
            out.append(self.mu(dot_id, N=N + e_hat, V_v=V_v))

        out = np.array(out).flatten()
        return out

    def triplepoint_h(
        self,
        new_voltages: Sequence[float],
        v_node_idx: Sequence[int],
        N: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Calculate chemical potentials of all dots for charge configuration
            N, corresponding to hole triple points, mu_j(N + e_hat_i)
        Args:
            new_voltages: values of new gate voltages, to be replaced in
                          self.V_v. These are the values scipy.optimize.fsolve
                          is solving for
            v_node_idx: Voltages nodes indexes to which the values above
                        correspond to.
            N: Desired charge configuration, if none supplied self.N is taken.
        """

        if N is None:
            N = np.array(self.N())
        V_v = np.array(self.V_v())
        V_v[v_node_idx] = new_voltages

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

        out = np.array(out).flatten()
        return out

    def mu(
        self,
        dot_indx: int,
        N: Optional[Sequence[int]] = None,
        V_v: Optional[Sequence[float]] = None,
    ) -> float:

        """The chemical potential of dot dot_id for a given charge and voltage
        configuration.

        Args:
            N: Number of electrons in each dot
            V_v: voltages node (gate) voltages

        Returns:
            chemical potential value
        """

        if N is None:
            N = np.array(self.N())

        if V_v is None:
            V_v = np.array(self.V_v())

        C_cc = np.array(self.C_cc())
        C_cv = np.array(self.C_cv())

        e_hat = np.zeros(len(N)).astype(int)
        e_hat[dot_indx] = 1

        pot = -(elem_charge ** 2) / 2 * multi_dot([e_hat, inv(C_cc), e_hat])
        pot += (elem_charge ** 2) * multi_dot([N, inv(C_cc), e_hat])
        pot += elem_charge * multi_dot([e_hat, inv(C_cc), C_cv, V_v])

        return pot

    def sweep_voltages(
        self,
        v_node_idx: Sequence[int],  # the one we want to sweep
        v_ranges: Sequence[Tuple[float, float]],
        n_steps: Sequence[int] = N_2D,
        line_intensity: float = 1.0,
        broaden: bool = True,
        add_noise: bool = True,
        kernel_widths: Tuple[float, float] = (1.0, 1.0),
        target_snr_db: float = 10,
        e_temp: float = 1e-20,
        normalize: bool = True,
        single_dot: bool = False,
        known_quality: Optional[int] = None,
        add_charge_jumps: bool = False,
        jump_freq: float = 0.001,
    ) -> Optional[int]:
        """
        Determine signal peaks by computing the energy
        """
        self.set_voltage(v_node_idx[0], np.min(v_ranges[0]))
        self.set_voltage(v_node_idx[1], np.min(v_ranges[1]))
        N_old = self.determine_N()

        voltage_x = np.linspace(
            np.min(v_ranges[0]), np.max(v_ranges[0]), n_steps[0]
        )

        voltage_y = np.linspace(
            np.min(v_ranges[1]), np.max(v_ranges[1]), n_steps[1]
        )
        signal = np.zeros(n_steps)

        if add_charge_jumps:
            x = np.ones(n_steps)
            s = 1
            trnsp = np.random.randint(2, size=1)

            poisson = np.random.poisson(lam=jump_freq, size=n_steps)
            poisson[poisson > 1] = 1
            for ix in range(n_steps[0]):
                for iy in range(n_steps[1]):
                    if poisson[ix, iy] == 1:
                        s *= -1
                    x[ix, iy] *= s
            x = np.array(x)
            x = (x + 1) / 2
        else:
            x = np.zeros(n_steps)

        for ivx, x_val in enumerate(voltage_x):
            self.set_voltage(v_node_idx[1], voltage_y[0])
            # N_old = self.determine_N()
            self.set_voltage(v_node_idx[0], x_val)
            for ivy, y_val in enumerate(voltage_y):
                self.set_voltage(v_node_idx[1], y_val)
                N_curr = self.determine_N()
                n_idx = int(np.random.randint(len(N_curr), size=1))
                # add charge jumps if desired:
                N_curr[n_idx] += x[ivx, ivy]
                self.N(N_curr)

                n_degen = self.get_number_of_degeneracies(
                    N_current=N_curr,
                    e_temp=e_temp,
                )
                if single_dot:
                    n_degen = np.min([1, n_degen])

                signal[ivx, ivy] = n_degen * line_intensity

        xm, ym = np.meshgrid(voltage_x, voltage_y)
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

        if single_dot:
            regime = "singledot"
        else:
            regime = "doubledot"
        dataid = self._save_to_db(
            [self.voltage_nodes[v_node_idx[0]].v,
             self.voltage_nodes[v_node_idx[1]].v],
            [voltage_x, voltage_y],
            signal,
            nt_label=[regime],
            quality=quality,
        )

        return dataid

    def sweep_voltage(
        self,
        v_node_idx: int,  # the one we want to sweep
        v_range: Sequence[float],
        n_steps: int = N_1D[0],
        line_intensity: float = 1.0,
        e_temp: float = 1e-20,
        kernel_width: Sequence[float] = [1.0],
        target_snr_db: float = 10.0,
        normalize: bool = True,
    ) -> Optional[int]:
        """
        Use self.determine_N to detect charge transitions
        """
        self.set_voltage(v_node_idx, np.min(v_range))
        signal = np.zeros(n_steps)

        voltage_x = np.linspace(np.min(v_range), np.max(v_range), n_steps)
        for iv, v_val in enumerate(voltage_x):
            self.set_voltage(v_node_idx, v_val)
            N_current = self.determine_N()
            self.N(N_current)

            n_degen = self.get_number_of_degeneracies(
                N_current=N_current,
                e_temp=e_temp,
            )

            signal[iv] = n_degen * line_intensity

        signal = self._make_it_real(signal, kernel_widths=kernel_width)
        signal = self._add_noise(signal, target_snr_db=target_snr_db)

        if normalize:
            signal = signal / np.max(signal)

        dataid = self._save_to_db(
            [self.voltage_nodes[v_node_idx].v],
            [voltage_x], signal, nt_label=["coulomboscillation"],
        )

        return dataid

    def get_number_of_degeneracies(
        self,
        N_current: Optional[Sequence[int]] = None,
        V_v: Optional[Sequence[float]] = None,
        e_temp: float = 1e-21,
    ) -> int:
        """
        Excludes chareg states with negative number of charges
        """
        n_dots = len(self.charge_nodes)

        if N_current is None:
            N_current = self.N()

        if V_v is None:
            V_v = self.V_v()

        current_energy = self.compute_energy(N=N_current, V_v=V_v)
        I_mat = np.eye(n_dots)
        energies = []
        c_configs = []

        def append_energy(charge_stage):
            if not np.any(np.array(charge_stage) < 0):
                energies.append(self.compute_energy(N=charge_stage, V_v=V_v))
                c_configs.append(charge_stage)

        for dot_id in range(n_dots):
            e_hat = I_mat[dot_id]

            append_energy(N_current + e_hat)
            append_energy(N_current - e_hat)

            for other_dot in range(n_dots):
                if other_dot != dot_id:
                    e_hat_other = I_mat[other_dot]
                    append_energy(N_current + e_hat - e_hat_other)

        current_energy = np.array([current_energy] * len(energies))
        dU = abs(np.array(energies) - current_energy)

        n_degen = np.sum(
            np.isclose(dU, np.zeros(len(dU)), atol=e_temp).astype(int)
        )

        return n_degen

    def _make_it_real(
        self,
        diagram: np.ndarray,
        kernel_widths: Sequence[float] = [3.0, 3.0],
    ) -> np.ndarray:
        """Make the tickfigure diagram more real by convolving it with a
        Gaussian. Currently for 2D diagrams only.

        Args:
            diagram: The previously computed stickfigure diagram
            kernel_width: Width of the Gaussian kernel.

        Return:
            Gaussian blurred diagram.
        """
        org_shape = diagram.shape
        diagram = gaussian_filter(
            diagram, sigma=kernel_widths[0], mode="constant", truncate=1
        )

        return resize(diagram, org_shape)

    def _add_noise(
        self,
        diagram: np.ndarray,
        target_snr_db: float = 10.0,
    ) -> np.ndarray:
        """Add noise to charge diagram to match the desired signal to noise
        ratio.

        Args:
            diagram: Noise free diagram
            target_snr_db: Target signal to noise ratio in dB
        Return:
            Noisy diagram
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
        data: np.ndarray,
        nt_label: Sequence[str],
        quality: int = 1,
        write_period: int = 10,
    ) -> Union[None, int]:
        """ Save data to database. Returns run id. """

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

        meta_add_on = dict.fromkeys(nt.config["core"]["meta_fields"], None)
        meta_add_on["device_name"] = self.name
        nm = dict.fromkeys(["dc_current", "dc_sensor", "rf"], (0, 1))
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
        v_node_idx: Sequence[int],
        V_v: Optional[Sequence[float]] = None,
        N_limits: Optional[N_lmt_type] = None,
    ) -> List[List[float]]:
        """
        Determine N by minimizing energy
        """
        if V_v is None:
            V_v = self.V_v()
        if N_limits is None:
            N_limits = [(0, 1)] * len(self.N())  # [(0, 1), (0, 1)]

        N_init = []
        for did in range(len(N_limits)):
            N_init.append(N_limits[did][0])

        def eng_sub_fct(N, swept_voltages):
            curr_V = V_v
            for iv, v_to_sweep in enumerate(v_node_idx):
                curr_V[v_to_sweep] = swept_voltages[iv]
            return self.compute_energy(N=N, V_v=curr_V)

        eng_fct = partial(eng_sub_fct, N_init)
        x0 = []
        for v_to_sweep in v_node_idx:
            x0.append(V_v[v_to_sweep])
        res = sc.optimize.minimize(
            eng_fct,
            x0,
            #    method='COBYLA',
            #    method='Powell',
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
            #    method='COBYLA',
            #    method='Powell',
            method="Nelder-Mead",
            tol=1e-4,
        )
        V_stop_config = res.x

        V_limits = [
            [V_init_config[0], V_stop_config[0]],
            [V_init_config[1], V_stop_config[1]],
        ]

        return V_limits



