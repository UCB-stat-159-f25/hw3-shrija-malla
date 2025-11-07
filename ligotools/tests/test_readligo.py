import numpy as np
from ligotools import readligo

def test_loaddata_output_structure():
    fn = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"
    strain, time, chan_dict = readligo.loaddata(fn, 'H1')
    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert isinstance(chan_dict, dict)

def test_loaddata_strain_not_empty():
    fn = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"
    strain, time, chan_dict = readligo.loaddata(fn, 'H1')
    assert len(strain) > 0
    assert len(time) == len(strain)

