from gwosc.datasets import find_datasets, event_gps
from gwosc import datasets

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




import json

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
    
with open('test.json', 'w') as f:
    json.dump(configs, f, ensure_ascii=False, indent=4)