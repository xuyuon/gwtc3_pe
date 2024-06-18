import jax.numpy as jnp
import jax

import numpy as np
import corner
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import os

from pandas import DataFrame



############################## Plot Posterior Samples ##############################
def plotPosterior(result, event, output_dir="output"):
    # labels = ["M_c", "eta", "s_1_z", "s_2_z", "dL", "t_c", "phase_c", "iota", "psi", "ra", "dec"]
    labels = ["M_c", "eta", "s_1_i", "s_1_j", "s_1_k", "s_2_i", "s_2_j", "s_2_k", "dL", "t_c", "phase_c", "iota", "psi", "ra", "dec"]
    
    samples = np.array(list(result.values())).reshape(15, -1) # flatten the array
    transposed_array = samples.T # transpose the array
    figure = corner.corner(transposed_array, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True)
    mkdir(output_dir + "/posterior_plot")
    plt.savefig(output_dir + "/posterior_plot/"+event+".jpeg")


############################## Save Posterior Samples ##############################
def savePosterior(result, event, output_dir="output"):
    samples = np.array(list(result.values())).reshape(15, -1) # flatten the array
    transposed_array = samples.T # transpose the array
    mkdir(output_dir + "/posterior_samples")
    with h5py.File(output_dir + '/posterior_samples/' + event + '.h5', 'w') as f:
        f.create_dataset('posterior', data=transposed_array)


def plotRunAnalysis(summary, event, output_dir="output"):
    chains, log_prob, local_accs, global_accs, loss_vals = summary.values()
    rng_key = jax.random.PRNGKey(42)
    rng_key, subkey = jax.random.split(rng_key)
    
    chains = np.array(chains)
    loss_vals = np.array(loss_vals)
    log_prob = np.array(log_prob)
    
    # Plot one chain to show the jump
    plt.figure(figsize=(6, 6))
    axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
    plt.sca(axs[0])
    plt.title("2 chains")
    plt.plot(chains[0, :, 0], chains[0, :, 1], alpha=0.5)
    plt.plot(chains[1, :, 0], chains[1, :, 1], alpha=0.5)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.sca(axs[1])
    plt.title("NF loss")
    plt.plot(loss_vals.reshape(-1))
    plt.xlabel("iteration")

    plt.sca(axs[2])
    plt.title("Local Acceptance")
    plt.plot(local_accs.mean(0))
    plt.xlabel("iteration")

    plt.sca(axs[3])
    plt.title("Global Acceptance")
    plt.plot(global_accs.mean(0))
    plt.xlabel("iteration")
    plt.tight_layout()
    
    mkdir(output_dir + "/posterior_analysis")
    plt.savefig(output_dir + "/posterior_analysis/"+event+".jpeg")

def plotLikelihood(summary, event, output_dir="output"):
    chains, log_prob, local_accs, global_accs, loss_vals = summary.values()
    log_prob = np.array(log_prob)
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    fig = plt.plot(log_prob[0], linewidth=1.0)
    plt.ylim(bottom=-20)
    mkdir(output_dir + "/likelihood_single_line")
    plt.savefig(output_dir + "/likelihood_single_line/"+event+".jpeg")

    # Plot each line
    for i in range(log_prob.shape[0]):
        ax.plot(log_prob[i], linewidth=0.05)
    plt.ylim(bottom=-20)
    mkdir(output_dir + "/likelihood")
    plt.savefig(output_dir + "/likelihood/"+event+".jpeg")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

"""
['final_spin', 'spin_2y', 'final_mass_source', 'spin_1y', 'cos_tilt_2', 'mass_1_source', 
'viewing_angle', 'spin_2x', 'inverted_mass_ratio', 'phi_2', 'chi_p', 'chirp_mass', 'chirp_mass_source', 
'total_mass', 'redshift', 'luminosity_distance', 'theta_jn', 'chi_eff', 'a_1', 'cos_iota', 
'radiated_energy', 'cos_theta_jn', 'total_mass_source', 'phi_jl', 'mass_2', 'ra', 'final_mass', 
'spin_1x', 'log_likelihood', 'tilt_2', 'tilt_1', 'psi', 'dec', 'symmetric_mass_ratio', 'mass_2_source', 
'iota', 'psi_J', 'cos_tilt_1', 'phi_12', 'mass_ratio', 'comoving_distance', 'phase', 'beta', 'chi_p_2spin', 
'phi_1', 'a_2', 'spin_1z', 'peak_luminosity', 'spin_2z', 'mass_1', 'tilt_1_infinity_only_prec_avg', 
'tilt_2_infinity_only_prec_avg', 'spin_1z_infinity_only_prec_avg', 'spin_2z_infinity_only_prec_avg', 
'chi_eff_infinity_only_prec_avg', 'chi_p_infinity_only_prec_avg', 'cos_tilt_1_infinity_only_prec_avg', 
'cos_tilt_2_infinity_only_prec_avg']
"""
def compare_plot(event_name, params, output_dir="compare_plot"):
    """
    params: A list of params to be included in the plot
    """
    bilby_posterior_dir = "data/IGWN-GWTC3p0-v1-" + event_name[:-3] + "_PEDataRelease_mixed_cosmo.h5"
    jim_posterior_dir = "output/posterior_samples/" + event_name + ".h5"    

    file = h5py.File(jim_posterior_dir, 'r')
    jim_posterior = np.array(file['posterior'])
    file.close()
    
    file = h5py.File(bilby_posterior_dir, 'r')
    
    jim_params = []
    bilby_params = []
    if "chirp_mass" in params:
        jim_params.append(jim_posterior[:,0])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['chirp_mass']))
    
    if "eta" in params:
        jim_params.append(jim_posterior[:,1])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['symmetric_mass_ratio']))
    
    if "spin1x" in params:
        jim_params.append(jim_posterior[:,2])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['spin_1x']))
    
    if "spin1y" in params:
        jim_params.append(jim_posterior[:,3])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['spin_1y']))
    
    if "spin1z" in params:
        jim_params.append(jim_posterior[:,4])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['spin_1z']))
    
    if "spin2x" in params:
        jim_params.append(jim_posterior[:,5])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['spin_2x']))
    
    if "spin2y" in params:
        jim_params.append(jim_posterior[:,6])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['spin_2y']))
    
    if "spin2z" in params:
        jim_params.append(jim_posterior[:,7])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['spin_2z']))
    
    if "luminosity_distance" in params:
        jim_params.append(jim_posterior[:,8])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['luminosity_distance']))
        
    if "phase" in params:
        jim_params.append(jim_posterior[:,10])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['phase']))
        
    if "iota" in params:
        jim_params.append(jim_posterior[:,11])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['iota']))
    
    if "psi" in params:
        jim_params.append(jim_posterior[:,12])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['psi']))
    
    if "ra" in params:
        jim_params.append(jim_posterior[:,13])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['ra']))
        
    if "dec" in params:
        jim_params.append(jim_posterior[:,14])
        bilby_params.append(np.array(file['C01:Mixed']['posterior_samples']['dec']))
    
    file.close()
        
    jim_params = np.array(jim_params).T
    sample_point_filter = np.random.choice(jim_params.shape[0], size=5000, replace=True)
    jim_params = jim_params[sample_point_filter]

    bilby_params = np.array(bilby_params).T
    sample_point_filter = np.random.choice(bilby_params.shape[0], size=5000, replace=True)
    bilby_params = bilby_params[sample_point_filter]
    # spin_1_theta = np.arccos(spin_1z)
    # spin_2_theta = np.arccos(spin_2z)
    # spin_1_phi = np.arctan2(spin_1y, spin_1x)
    # spin_2_phi = np.arctan2(spin_2y, spin_2x)
    # spin_1_r = np.sqrt(spin_1x**2 + spin_1y**2 + spin_1z**2)
    # spin_2_r = np.sqrt(spin_2x**2 + spin_2y**2 + spin_2z**2)


    labels = params

    fig = corner.corner(jim_params, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'red')
    corner.corner(bilby_params, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'blue', fig=fig)

    plt.savefig(output_dir+"/"+event_name+".jpeg")
    
    

def compare_intrinsic_params(event_name, output_dir="compare_plot"):
    compare_plot(event_name, ["chirp_mass", "eta", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"], output_dir)
  

def compare_extrinsic_params(event_name, output_dir="compare_plot"):
    compare_plot(event_name, ["luminosity_distance", "phase", "iota", "psi", "ra", "dec"], output_dir)


def plot_prior(event_name, params, output_dir="prior"):
    """
    params: A list of params to be included in the plot
    """
    bilby_posterior_dir = "data/IGWN-GWTC3p0-v1-" + event_name[:-3] + "_PEDataRelease_mixed_cosmo.h5"



def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two samples.
    Parameters
    ----------
    x : 1D np array
        Samples from distribution P, which typically represents the true
        distribution.
    y : 1D np rray
        Samples from distribution Q, which typically represents the approximate
        distribution.
    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
    """ 
    bins = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)), 100) 
    # The size of array must be much greater than the number of bins
    
    prob_x = np.histogram(x, bins=bins)[0]/x.shape
    prob_y = np.histogram(y, bins=bins)[0]/y.shape
       
    return np.sum(np.where((prob_y != 0) & (prob_x != 0), prob_y * (np.log(prob_y) - np.log(prob_x)), 0))
    
    
def test(x, y):
    return (1+(0.5)**2)/(2)-0.5 


def JSdivergence(P, Q):
  #create M
  M=(P+Q)/2 #sum the two distributions then get average

  kl_p_q = KLdivergence(P, M)
  kl_q_p = KLdivergence(Q, M)

  js = (kl_p_q+kl_q_p)/2
  return js


def output_summary(events):
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
    




