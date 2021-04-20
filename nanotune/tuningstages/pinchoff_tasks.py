# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


def update_gatecharacterization_ranges(
    self,
    range_update_directives: List[str],
) -> None:
    """"""
    for directive in range_update_directives:
        if directive not in ["x more negative", "x more positive"]:
            logger.error(
                (f'{self.stage}: Unknown range update directive.'
                'Cannot update measurement setting')
            )

    if "x more negative" in range_update_directives:
        self._update_range(0, 0)
    if "x more positive" in range_update_directives:
        self._update_range(0, 1)


def _update_range(self, gate_id, range_id):
    v_change = abs(
        self.current_ranges[gate_id][range_id]
        - self.gate.safety_range()[range_id]
    )
    sign = (-1) ** (range_id + 1)
    self.current_ranges[gate_id][range_id] += sign * v_change

