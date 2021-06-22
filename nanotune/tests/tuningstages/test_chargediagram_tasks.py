# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
from nanotune.tuningstages.chargediagram_tasks import *
from nanotune.fit.dotfit import DotFit
from nanotune.tests.mock_classifier import MockClassifer

def test_segment_dot_data(db_dot_tuning, tmp_path):
    dot_segments = segment_dot_data(
        1001,
        'dot_tuning_data.db',
        tmp_path,
        segment_db_name="temp.db",
        segment_db_folder=tmp_path,
        segment_size=0.1,
    )
    ds = nt.Dataset(1, "temp.db", db_folder=tmp_path)
    assert bool(ds.data.transport.voltage_x[0] == dot_segments[1]['voltage_ranges'][0][0])
    assert bool(ds.data.transport.voltage_y[0] == dot_segments[1]['voltage_ranges'][1][0])

    dot_segments = segment_dot_data(
        1001,
        'dot_tuning_data.db',
        tmp_path,
        segment_size=0.1,
    )
    ds = nt.Dataset(1, "segmented_dot_tuning_data.db", db_folder=tmp_path)
    assert bool(ds.data.transport.voltage_x[0] == dot_segments[1]['voltage_ranges'][0][0])
    assert bool(ds.data.transport.voltage_y[0] == dot_segments[1]['voltage_ranges'][1][0])

def test_classify_dot_segments(db_dot_tuning, tmp_path):
    df = DotFit(
        1001, 'dot_tuning_data.db',
        db_folder=tmp_path, segment_size = 0.1
    )
    dot_segments = df.save_segmented_data_return_info(
        'segmented_dot_tuning_data.db', segment_db_folder=tmp_path)
    classifiers={
        'singledot': MockClassifer('singledot'),
        'doubledot': MockClassifer('doubledot'),
        'dotregime': MockClassifer('dotregime'),
    }

    clf_result = classify_dot_segments(
        classifiers,
        list(dot_segments.keys()),
        'segmented_dot_tuning_data.db',
        db_folder=tmp_path,
    )
    assert sorted(list(clf_result.keys())) == [1, 2, 3, 4]