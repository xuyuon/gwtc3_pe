import jax.numpy as jnp
import jax

import numpy as np
import corner
from pathlib import Path
import matplotlib.pyplot as plt
import h5py



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

def plotLikelihood(summary, event):
    chains, log_prob, local_accs, global_accs, loss_vals = summary.values()
    log_prob = np.array(log_prob)
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    fig = plt.plot(log_prob[0], linewidth=0.5)
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