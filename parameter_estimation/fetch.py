from gwosc.datasets import find_datasets, event_gps
from gwosc import datasets
import h5py
import numpy as np

from .alias import JIM_TO_BILBY_ALIAS


def fetch_gps(event):
    """
    input:
        event: "GW150914"
    """
    return event_gps(event)


def fetch_detectors(event):
    """
    input:
        event: "GW150914"
    """
    detectors = []
    if event in datasets.find_datasets(type='events', detector='H1'):
        detectors.append('H1')
    if event in datasets.find_datasets(type='events', detector='L1'):
        detectors.append('L1')
    if event in datasets.find_datasets(type='events', detector='V1'):
        detectors.append('V1')
    return detectors
    

def fetch_catalog(event):
    """
    input:
        event: "GW150914"
    """
    if event in datasets.find_datasets(type='events', catalog='GWTC-1-confident'):
        return 'GWTC-1-confident'
    if event in datasets.find_datasets(type='events', catalog='GWTC-2.1-confident'):
        return 'GWTC-2.1-confident'
    if event in datasets.find_datasets(type='events', catalog='GWTC-3-confident'):
        return 'GWTC-3-confident'
    

def getBilbyPosterior(event, param, directory="data"):
    """
    Get the bilby posterior sample points from a h5py file
    """
    bilby_posterior_dir = directory+"/IGWN-GWTC3p0-v1-" + event[:-3] + "_PEDataRelease_mixed_cosmo.h5"
    with h5py.File(bilby_posterior_dir) as f:
        return np.array(f['C01:Mixed']['posterior_samples'][JIM_TO_BILBY_ALIAS[param]])

