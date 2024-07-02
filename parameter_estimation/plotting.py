import corner
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import h5py

from .utilities import mkdir
from .fetch import getBilbyPosterior
from .save import getPosterior

"""
Available params:
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

############################## Functions for Plotting Various Graphs ##############################

def plotPosterior(result, event, output_dir="output", labels=["M_c", "eta", "s_1_i", "s_1_j", "s_1_k", "s_2_i", "s_2_j", "s_2_k", "dL", "t_c", "phase_c", "iota", "psi", "ra", "dec"]):
    """
    Plot the posterior samples in a corner plot
    """
    
    samples = np.array(list(result.values())).reshape(int(len(labels)), -1) # flatten the array
    transposed_array = samples.T # transpose the array
    figure = corner.corner(transposed_array, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True)
    mkdir(output_dir + "/posterior_plot")
    plt.savefig(output_dir + "/posterior_plot/"+event+".jpeg")
    
    
def plotRunningProgress(summary, event, output_dir="output"):
    """
    Plot how the parameter estimation run progresses
    """
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
    """
    Plot the likelihood of the run over epochs
    """
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
    
    
def plotCompare(event_name, params, output_dir="compare_plot"):
    """
    To compare the posterior samples from Jim and Bilby
    
    params: A list of params to be included in the plot
    """
    jim_params = []
    bilby_params = []
    for param in params:
        jim_params.append(getPosterior(event_name, param))
        bilby_params.append(getBilbyPosterior(event_name, param))
        
    jim_params = np.array(jim_params).T
    sample_point_filter = np.random.choice(jim_params.shape[0], size=5000, replace=True)
    jim_params = jim_params[sample_point_filter]

    bilby_params = np.array(bilby_params).T
    sample_point_filter = np.random.choice(bilby_params.shape[0], size=5000, replace=True)
    bilby_params = bilby_params[sample_point_filter]

    fig = corner.corner(jim_params, labels=params, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'red')
    corner.corner(bilby_params, labels=params, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'blue', fig=fig)

    plt.savefig(output_dir+"/"+event_name+".jpeg")
    
    
def plotIntrinsicParamsComparison(event_name, output_dir="compare_plot"):
    """
    To compare the intrinsic parameters of the posterior samples from Jim and Bilby
    """
    # plotCompare(event_name, ["chirp_mass", "eta", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"], output_dir)
    plotCompare(event_name, ["M_c", "eta", "a_1", "a_2", "phi_12", "phi_jl", "tilt_1", "tilt_2"], output_dir)

def plotExtrinsicParamsComparison(event_name, output_dir="compare_plot"):
    """
    To compare the extrinsic parameters of the posterior samples from Jim and Bilby
    """
    # plotCompare(event_name, ["luminosity_distance", "phase", "iota", "psi", "ra", "dec"], output_dir)
    plotCompare(event_name, ["d_L", "phase_c", "theta_jn", "psi", "ra", "dec"], output_dir)