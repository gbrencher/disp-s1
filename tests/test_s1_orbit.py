"""Synthetic S1 orbit metadata: filenames must not alter the orbit."""
from datetime import datetime, timedelta
import h5py
import numpy as np
import pytest
import isce3
from disp_s1._baselines import get_cslc_orbit


def test_compressed_filename_orbit(tmp_path):
    path = tmp_path / 'compressed_t012_023790_iw1_20240428_20231112_20240428.h5'
    times = np.arange(12, dtype=float) * 10
    positions = np.column_stack([np.full(12, 7000000.), times * 7500, np.zeros(12)])
    velocities = np.tile([0., 7500., 0.], (12, 1))
    epoch = datetime(2024, 4, 28)
    with h5py.File(path, 'w') as f:
        g = f.create_group('metadata/orbit')
        g['time'] = times
        g['reference_epoch'] = np.bytes_(epoch.isoformat())
        for i, axis in enumerate('xyz'):
            g['position_' + axis] = positions[:, i]
            g['velocity_' + axis] = velocities[:, i]
    # Independently reproduce the pinned upstream state-vector conversion.
    expected = isce3.core.Orbit([
        isce3.core.StateVector(isce3.core.DateTime(epoch + timedelta(seconds=t)), x, v)
        for t, x, v in zip(times, positions, velocities)])
    actual = get_cslc_orbit(path)
    for t in (20., 55., 90.):
        for a, b in zip(actual.interpolate(t), expected.interpolate(t)):
            np.testing.assert_array_equal(a, b)
    renamed = tmp_path / 'arbitrary-local-name.h5'
    path.rename(renamed)
    np.testing.assert_array_equal(get_cslc_orbit(renamed).interpolate(55.)[0], actual.interpolate(55.)[0])


def test_missing_metadata_is_not_silently_accepted(tmp_path):
    path = tmp_path / 'compressed_invalid.h5'
    with h5py.File(path, 'w'):
        pass
    with pytest.raises(KeyError):
        get_cslc_orbit(path)
