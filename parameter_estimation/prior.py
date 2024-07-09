from jimgw.prior import Composite, Unconstrained_Uniform, Sphere, UniformInComponentChirpMass, UniformInComponentMassRatio, PowerLaw
import numpy as np
import jax.numpy as jnp
import corner
import matplotlib.pyplot as plt
import jax

from .utilities import spin_to_spin
from .alias import JIM_TO_BILBY_ALIAS

def prior_setting_1(M_c):
    Mc_prior = UniformInComponentChirpMass(M_c[0], M_c[1], naming=["M_c"])
    q_prior = UniformInComponentMassRatio(
        0.125, 
        1.0, 
        naming=["q"],
        transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
    )
    s1_prior = Sphere(naming="s1")
    s2_prior = Sphere(naming="s2")
    dL_prior = PowerLaw(100.0, 10000.0, 2.0, naming=["d_L"])
    t_c_prior = Unconstrained_Uniform(-0.5, 0.5, naming=["t_c"])
    phase_c_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
    cos_iota_prior = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["cos_iota"],
        transforms={
            "cos_iota": (
                "iota",
                lambda params: jnp.arccos(
                    jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )
    psi_prior = Unconstrained_Uniform(0.0, jnp.pi, naming=["psi"])
    ra_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["ra"])
    sin_dec_prior = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["sin_dec"],
        transforms={
            "sin_dec": (
                "dec",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )
    
    prior = Composite(
        [
            Mc_prior,
            q_prior,
            s1_prior,
            s2_prior,
            dL_prior,
            t_c_prior,
            phase_c_prior,
            cos_iota_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
        ],
    )
    return prior

def prior_setting_2(M_c):
    Mc_prior = UniformInComponentChirpMass(M_c[0], M_c[1], naming=["M_c"])
    q_prior = UniformInComponentMassRatio(
        0.125, 
        1.0, 
        naming=["q"],
        transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
    )
    a1_prior = Unconstrained_Uniform(0.0, 0.99, naming=["a_1"])
    a2_prior = Unconstrained_Uniform(0.0, 0.99, naming=["a_2"])
    phi_12_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["phi_12"])
    phi_jl_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["phi_jl"])
    sin_theta_jn = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["sin_theta_jn"],
        transforms={
            "sin_theta_jn": (
                "theta_jn",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_theta_jn"] / 2 * jnp.pi)) * 2 / jnp.pi
                )+jnp.pi/2,
            )
        },
    )
    sin_tilt_1 = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["sin_tilt_1"],
        transforms={
            "sin_tilt_1": (
                "tilt_1",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_tilt_1"] / 2 * jnp.pi)) * 2 / jnp.pi
                )+jnp.pi/2,
            )
        },
    )
    sin_tilt_2 = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["sin_tilt_2"],
        transforms={
            "sin_tilt_2": (
                "tilt_2",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_tilt_2"] / 2 * jnp.pi)) * 2 / jnp.pi
                )+jnp.pi/2,
            )
        },
    )
    dL_prior = PowerLaw(100.0, 10000.0, 2.0, naming=["d_L"])
    t_c_prior = Unconstrained_Uniform(-0.5, 0.5, naming=["t_c"])
    phase_c_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
    psi_prior = Unconstrained_Uniform(0.0, jnp.pi, naming=["psi"])
    ra_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["ra"])
    sin_dec_prior = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["sin_dec"],
        transforms={
            "sin_dec": (
                "dec",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )

    prior = Composite(
        [
            Mc_prior,
            q_prior,
            a1_prior,
            a2_prior,
            phi_12_prior,
            phi_jl_prior,
            sin_theta_jn,
            sin_tilt_1,
            sin_tilt_2,
            dL_prior,
            t_c_prior,
            phase_c_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
        ],
    )
    return prior

def testing(composite_prior):
    import bilby
    
    # Bilby priors
    bilby_priors = bilby.core.prior.PriorDict()
    bilby_priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(3.0, 30.0, name='chirp_mass')
    bilby_priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(0.125, 1.0, name='mass_ratio')
    bilby_priors['a_1'] = bilby.prior.Uniform(0.0, 0.99, name='a_1')
    bilby_priors['a_2'] = bilby.prior.Uniform(0.0, 0.99, name='a_2')
    bilby_priors['phi_12'] = bilby.prior.Uniform(0.0, 2*np.pi, name='phi_12')
    bilby_priors['phi_jl'] = bilby.prior.Uniform(0.0, 2*np.pi, name='phi_jl')
    bilby_priors['theta_jn'] = bilby.core.prior.analytical.Sine(0.0, np.pi, name='theta_jn')
    bilby_priors['tilt_1'] = bilby.core.prior.analytical.Sine(0.0, np.pi, name='tilt_1')
    bilby_priors['tilt_2'] = bilby.core.prior.analytical.Sine(0.0, np.pi, name='tilt_2')
    bilby_priors['luminosity_distance'] = bilby.core.prior.analytical.PowerLaw(2.0, 100.0, 10000.0, name='luminosity_distance')
    bilby_priors['geocent_time'] = bilby.prior.Uniform(-0.5, 0.5, name='geocent_time')
    bilby_priors['phase'] = bilby.prior.Uniform(0.0, 2*np.pi, name='phase')
    bilby_priors['psi'] = bilby.prior.Uniform(0.0, np.pi, name='psi')
    bilby_priors['ra'] = bilby.prior.Uniform(0.0, 2*np.pi, name='ra')
    bilby_priors['dec'] = bilby.core.prior.analytical.Cosine(name='dec')
    
    bilby_samples = bilby_priors.sample(5000)
    
    bilby_samples['mass_1'], bilby_samples['mass_2'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(bilby_samples['chirp_mass'], bilby_samples['mass_ratio'])
    bilby_samples['symmetric_mass_ratio'] = bilby.gw.conversion.component_masses_to_symmetric_mass_ratio(bilby_samples['mass_1'], bilby_samples['mass_2'])
    bilby_samples['iota'], bilby_samples['spin_1x'], bilby_samples['spin_1y'], bilby_samples['spin_1z'], bilby_samples['spin_2x'], bilby_samples['spin_2y'], bilby_samples['spin_2z'] = spin_to_spin(bilby_samples['theta_jn'], bilby_samples['phi_jl'], bilby_samples['tilt_1'], bilby_samples['tilt_2'], bilby_samples['phi_12'], bilby_samples['a_1'], bilby_samples['a_2'], bilby_samples['chirp_mass'], bilby_samples['symmetric_mass_ratio'], 20, bilby_samples['phase'])

    prior_list = composite_prior.priors
    jim_samples = composite_prior.sample(jax.random.PRNGKey(42), 5000)
    # transform the prior
    jim_samples = composite_prior.transform(composite_prior.add_name(jim_samples.values()))
    
    bilby_samples_selected = []
    labels = []
    for param in jim_samples:
        bilby_samples_selected.append(bilby_samples[JIM_TO_BILBY_ALIAS[param]])
        labels.append(param)
    
    jim_samples = np.array([x for x in jim_samples.values()]).T
    bilby_samples_selected = np.array(bilby_samples_selected).T
        
    fig = corner.corner(jim_samples, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'red')
    corner.corner(bilby_samples_selected, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True, color = 'blue', fig=fig)

    plt.savefig("priors_1.jpeg")
    plt.close()
    

