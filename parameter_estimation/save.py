import numpy as np
import h5py
from pandas import DataFrame
import json

from gwosc.datasets import find_datasets, event_gps
from gwosc import datasets

from .utilities import mkdir, JSdivergence
from .fetch import fetch_gps, fetch_detectors, fetch_catalog

############################## Save Posterior Samples ##############################
def savePosterior(result, event, output_dir="output"):
    """
    Save the posterior sample points into a h5py file
    keys:
    ----------
        posterior: posterior sample points
        parameters: list of parameters contained in the file
    """
    mkdir(output_dir + '/posterior_samples')
    with h5py.File(output_dir + '/posterior_samples/' + event + '.h5', 'w') as f:
        f.create_dataset('parameters', data=list(result.keys()))
        for param in result.keys():
            f.create_dataset('posterior/'+param, data=result[param])


def getPosterior(event, param, output_dir="output"):
    """
    Get the posterior sample points from a h5py file
    """
    with h5py.File(output_dir + '/posterior_samples/' + event + '.h5', 'r') as f:
        return np.array(f['posterior/'+param]).reshape(-1)
    

############################## Save Summary of PE Results ##############################
def saveSummary(events):
    """
    event: a list of events
    """
    
    chirp_mass = []
    standard_chirp_mass = []
    chirp_mass_JS = []
    
    eta = []
    standard_eta = []
    eta_JS = []
    
    luminosity_distance = []
    standard_luminosity_distance = []
    luminosity_distance_JS = []
    
    for event in events:
        print("Generating summary for "+event)
        bilby_posterior_dir = "data/IGWN-GWTC3p0-v1-" + event[:-3] + "_PEDataRelease_mixed_cosmo.h5"
        jim_posterior_dir = "output/posterior_samples/" + event + ".h5"    

        file = h5py.File(jim_posterior_dir, 'r')
        jim_posterior = np.array(file['posterior'])
        file.close()

        file = h5py.File(bilby_posterior_dir, 'r')
        
        jim_chirp_mass = jim_posterior[:,0]
        chirp_mass.append(np.mean(jim_chirp_mass))
        standard_chirp_mass.append(np.mean(np.array(file['C01:Mixed']['posterior_samples']['chirp_mass'])))
        chirp_mass_JS.append(JSdivergence(np.random.choice(jim_chirp_mass, size=10000), np.random.choice(np.array(file['C01:Mixed']['posterior_samples']['chirp_mass']), size=10000)))
        
        jim_eta = jim_posterior[:,1]
        eta.append(np.mean(jim_eta))
        standard_eta.append(np.mean(np.array(file['C01:Mixed']['posterior_samples']['symmetric_mass_ratio'])))
        eta_JS.append(JSdivergence(np.random.choice(jim_eta, size=10000), np.random.choice(np.array(file['C01:Mixed']['posterior_samples']['symmetric_mass_ratio']), size=10000)))
        
        jim_luminosity_distance = jim_posterior[:,8]
        luminosity_distance.append(np.mean(jim_luminosity_distance))
        standard_luminosity_distance.append(np.mean(np.array(file['C01:Mixed']['posterior_samples']['luminosity_distance'])))
        luminosity_distance_JS.append(JSdivergence(np.random.choice(jim_luminosity_distance, size=10000), np.random.choice(np.array(file['C01:Mixed']['posterior_samples']['luminosity_distance']), size=10000)))
        
        file.close()
    
    df = DataFrame({'event': events, 'chirp_mass': chirp_mass, 'standard_chirp_mass': standard_chirp_mass, 'chirp_mass_JS': chirp_mass_JS, 'eta': eta, 'standard_eta': standard_eta, 'eta_JS': eta_JS, 'luminosity_distance': luminosity_distance, 'standard_luminosity_distance': standard_luminosity_distance, 'luminosity_distance_JS': luminosity_distance_JS})
    df.to_excel('test.xlsx', sheet_name='pe_result', index=False)
    
    
############################## Generate config file ##############################
def generateDefaultConfig(file_name):
    """
    Generate a default config file for the PE
    """
    configs = {'configs': {}}

    for event in (datasets.find_datasets(type='events', catalog='GWTC-1-confident')+datasets.find_datasets(type='events', catalog='GWTC-2.1-confident')+datasets.find_datasets(type='events', catalog='GWTC-3-confident')):
        try:
            print("writing event" + event)
            data = {
                "event": event,
                "catalog": fetch_catalog(event),
                "gps": fetch_gps(event),
                "detectors": fetch_detectors(event),
                "duration": 4,
                "Mc_prior": {
                    "min": 1,
                    "max": 100
                },
            }
            configs['configs'][event] = data
        except Exception as e: print(e)
        
    with open(file_name, 'w') as f:
        json.dump(configs, f, ensure_ascii=False, indent=4)