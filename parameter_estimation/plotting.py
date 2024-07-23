import corner
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import h5py

from .utilities import mkdir, spin_to_spin, eta_to_q
from .fetch import getBilbyPosterior
from .save import getPosterior


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
    
    
def plotCompare(event_name, 
    params=["M_c", "q", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z", "d_L", "phase_c", "iota", "psi", "ra", "dec"], 
    output_dir="compare_plot", 
    result_dir="output", 
    result2_dir=None
):
    """
    To compare the posterior samples from Jim and Bilby
    
    params: A list of params to be included in the plot
    """
    jim_params = []
    bilby_params = []
    jim2_params = []
    for param in params:
        if param == "q":
            jim_params.append(eta_to_q(getPosterior(event_name, "eta", result_dir)))
            bilby_params.append(eta_to_q(getBilbyPosterior(event_name, "eta")))
        else:
            jim_params.append(getPosterior(event_name, param, result_dir))
            bilby_params.append(getBilbyPosterior(event_name, param))
    
    if result2_dir != None:
        spin_params = ['M_c', 'eta', 'theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2', 'd_L', 'phase_c', 'ra', 'dec', 'psi']
        for param in spin_params:
            jim2_params.append(getPosterior(event_name, param, result2_dir))
        iota, s1x, s1y, s1z, s2x, s2y, s2z = spin_to_spin(jim2_params[2], jim2_params[3], jim2_params[4], jim2_params[5], jim2_params[6], jim2_params[7], jim2_params[8], jim2_params[0], jim2_params[1], 20, jim2_params[13])
        jim2_params = [jim2_params[0], eta_to_q(jim2_params[1]), s1x, s1y, s1z, s2x, s2y, s2z, jim2_params[9], jim2_params[10], iota, jim2_params[13], jim2_params[11], jim2_params[12]]
    
    jim_params = np.array(jim_params).T
    sample_point_filter = np.random.choice(jim_params.shape[0], size=5000, replace=True)
    jim_params = jim_params[sample_point_filter]

    bilby_params = np.array(bilby_params).T
    sample_point_filter = np.random.choice(bilby_params.shape[0], size=5000, replace=True)
    bilby_params = bilby_params[sample_point_filter]
    
    if result2_dir != None:
        jim2_params = np.array(jim2_params).T
        sample_point_filter = np.random.choice(jim2_params.shape[0], size=5000, replace=True)
        jim2_params = jim2_params[sample_point_filter]
   
    # write a range for the corner plot using minimum value and maximum value of the data as limits
    param_range = []
    for i in range(len(params)):
        param_range.append([min(np.min(jim_params[:, i]), np.min(bilby_params[:, i])), max(np.max(jim_params[:, i]), np.max(bilby_params[:, i]))])
    fig = corner.corner(jim_params, bins=20, labels=params, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'red', range=param_range)
    corner.corner(bilby_params, bins=20, labels=params, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'blue', fig=fig, range=param_range)
    if result2_dir != None:
        corner.corner(jim2_params, bins=20,labels=params, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'orange', fig=fig, range=param_range)
    plt.savefig(output_dir+"/"+event_name+".jpeg")
    plt.close()
    
    
def plotIntrinsicParamsComparison(event_name, output_dir="compare_plot", result_dir="output"):
    """
    To compare the intrinsic parameters of the posterior samples from Jim and Bilby
    """
    # plotCompare(event_name, ["M_c", "eta", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z"], output_dir, result_dir)
    plotCompare(event_name, ["M_c", "eta", "a_1", "a_2", "phi_12", "phi_jl", "tilt_1", "tilt_2"], output_dir)

def plotExtrinsicParamsComparison(event_name, output_dir="compare_plot", result_dir="output"):
    """
    To compare the extrinsic parameters of the posterior samples from Jim and Bilby
    """
    # plotCompare(event_name, ["d_L", "phase_c", "iota", "psi", "ra", "dec"], output_dir, result_dir)
    plotCompare(event_name, ["d_L", "phase_c", "theta_jn", "psi", "ra", "dec"], output_dir)

# posterior_dir_list = ["GW191204_110529-v1", "GW191204_110529-v2", "GW191204_110529-v3", "GW191204_110529-v4", "GW191204_110529-v5", "GW191204_110529-v6", "GW191204_110529-v7", "GW191204_110529-v8", "GW191204_110529-v9", "GW191204_110529-v10"]
def plot10(posterior_dir_list, output_dir=None):
    import seaborn as sns
    colour_palette = sns.color_palette("magma", n_colors=10)
    fig = plt.figure(figsize=(30, 30))
    params=["M_c", "eta", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z", "d_L", "phase_c", "iota", "psi", "ra", "dec"]
    for posterior_dir, colour in zip(posterior_dir_list, colour_palette):
        jim_params = []
        for param in params:
            jim_params.append(getPosterior(posterior_dir, param, "output"))
        jim_params = np.array(jim_params).T
        sample_point_filter = np.random.choice(jim_params.shape[0], size=5000, replace=True)
        jim_params = jim_params[sample_point_filter]
        corner.corner(jim_params, labels=params, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = colour, fig=fig)
    plt.savefig("test.jpeg")
    plt.close()

event_name = "GW191204_110529-v1"
def plot11(posterior_dir_list=["GW191204_110529-v1", "GW191204_110529-v2", "GW191204_110529-v3", "GW191204_110529-v4", "GW191204_110529-v5", "GW191204_110529-v6", "GW191204_110529-v7", "GW191204_110529-v8", "GW191204_110529-v9", "GW191204_110529-v10"],
           output_dir=None):
    params=["M_c", "eta", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z", "d_L", "phase_c", "iota", "psi", "ra", "dec"]
    jim_params = []
    bilby_params = []
    for param in params:
        bilby_params.append(getBilbyPosterior(event_name, param))
    # put all the posterior samples from posterior_dir_list in jim_params
    for param in params:
        posteriors = np.array([])
        for posterior_dir in posterior_dir_list:
            posteriors = np.append(posteriors, getPosterior(posterior_dir, param, "output"))
        jim_params.append(posteriors)
    
    jim_params = np.array(jim_params).T
    sample_point_filter = np.random.choice(jim_params.shape[0], size=10000, replace=True)
    jim_params = jim_params[sample_point_filter]

    bilby_params = np.array(bilby_params).T
    sample_point_filter = np.random.choice(bilby_params.shape[0], size=10000, replace=True)
    bilby_params = bilby_params[sample_point_filter]
    
    fig = corner.corner(jim_params, labels=params, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'red')
    corner.corner(bilby_params, labels=params, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'blue', fig=fig)
    plt.savefig("test_3.jpeg")
    plt.close()
    
# posterior_dir_list = ["GW191204_110529-v1", "GW191204_110529-v2", "GW191204_110529-v3", "GW191204_110529-v4", "GW191204_110529-v5", "GW191204_110529-v6", "GW191204_110529-v7", "GW191204_110529-v8", "GW191204_110529-v9", "GW191204_110529-v10"]
def initializePoints(n_samples,
                     posterior_dir_list=["GW191204_110529-v1", "GW191204_110529-v2", "GW191204_110529-v3", "GW191204_110529-v4", "GW191204_110529-v5", "GW191204_110529-v6", "GW191204_110529-v7", "GW191204_110529-v8", "GW191204_110529-v9", "GW191204_110529-v10"],
                     output_dir=None
    ):
    params=["M_c", "eta", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z", "d_L", "t_c", "phase_c", "iota", "psi", "ra", "dec"]
    jim_params = []
    # put all the posterior samples from posterior_dir_list in jim_params
    for param in params:
        posteriors = np.array([])
        for posterior_dir in posterior_dir_list:
            posteriors = np.append(posteriors, getPosterior(posterior_dir, param, "output"))
        jim_params.append(posteriors)
    
    jim_params = np.array(jim_params).T
    sample_point_filter = np.random.choice(jim_params.shape[0], size=n_samples, replace=True)
    jim_params = jim_params[sample_point_filter]
    return jim_params

def initializeNF(posterior_dir_list=["GW191204_110529-v1", "GW191204_110529-v2", "GW191204_110529-v3", "GW191204_110529-v4", "GW191204_110529-v5", "GW191204_110529-v6", "GW191204_110529-v7", "GW191204_110529-v8", "GW191204_110529-v9", "GW191204_110529-v10"]):
    import jax
    import jax.numpy as jnp  # JAX NumPy
    import jax.random as random  # JAX random
    import optax  # Optimizers
    import equinox as eqx  # Equinox

    from flowMC.nfmodel.realNVP import RealNVP
    from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline

    import numpy as np

    params=["M_c", "eta", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z", "d_L", "t_c", "phase_c", "iota", "psi", "ra", "dec"]
    jim_params=[]
    for param in params:
        posteriors = np.array([])
        for posterior_dir in posterior_dir_list:
            posteriors = np.append(posteriors, getPosterior(posterior_dir, param, "output"))
        jim_params.append(posteriors)
    
    jim_params = np.array(jim_params).T
    sample_point_filter = np.random.choice(jim_params.shape[0], size=10000, replace=True)
    jim_params = jim_params[sample_point_filter]

    # Model parameters
    n_feature = 15
    n_layers = 10
    n_hiddens = [128, 128]
    n_bins = 8

    key, subkey = jax.random.split(jax.random.PRNGKey(1))

    model = MaskedCouplingRQSpline(
        n_feature,
        n_layers,
        n_hiddens,
        n_bins,
        subkey,
        data_cov=jnp.cov(jim_params.T),
        data_mean=jnp.mean(jim_params, axis=0),
    )

    num_epochs = 100
    batch_size = 10000
    learning_rate = 0.001
    momentum = 0.9

    optim = optax.adam(learning_rate)
    state = optim.init(eqx.filter(model, eqx.is_array))
    key, subkey = jax.random.split(key)
    key, model, state, loss = model.train(key, jim_params, optim, state, num_epochs, batch_size, verbose=True)
    return model