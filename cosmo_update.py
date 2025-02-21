#| Define LSBI inference function

##| This version uses the development version of lsbi to plot mixture models. This needs to be pip installed from the plot branch
## Initialise command line parameters: Nsim, resolution,shape and filename
import sys
## if no arguments are provided, use the default values
if len(sys.argv) < 5:
    Nsim = 10000
    n_runs = 4
    N_shape = 2
    filename = 'cosmo_update-10000-4-2.pdf'
else:
    Nsim = int(sys.argv[1])
    n_runs = int(sys.argv[2])
    N_shape = int(sys.argv[3])
    filename = sys.argv[4]+"-"+sys.argv[1]+"-"+sys.argv[2]+"-"+sys.argv[3]+".pdf"
print(Nsim, n_runs, N_shape, filename)

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import invwishart, matrix_normal, norm
from lsbi.model import MixtureModel
# np.random.seed(0)

def LSBI(θ, D, *args, **kwargs):
    shape = kwargs.pop('shape', ())
    if isinstance(shape, int):
        shape = (shape,)
    k, n = θ.shape
    d = D.shape[1]
    θD = np.concatenate([θ, D], axis=1)
    mean = θD.mean(axis=0)
    θbar = mean[:n]
    Dbar = mean[n:]

    cov = np.cov(θD.T)
    Θ = cov[:n, :n]
    Δ = cov[n:, n:]
    Ψ = cov[n:, :n]
    ν = k - d - n - 2
    invΘ = np.linalg.inv(Θ)

    C_ = invwishart(df=ν, scale=k*(Δ-Ψ @ invΘ @ Ψ.T)).rvs(shape)
    L1 = np.linalg.cholesky(C_/k)
    L2 = np.linalg.cholesky(invΘ)
    M_ = Ψ @ invΘ + np.einsum('...jk,...kl,ml->...jm', L1, np.random.randn(*shape, d, n), L2)
    m_ = Dbar - M_ @ θbar + np.einsum('...jk,...k->...j', L1, np.random.randn(*shape, d))
    return MixtureModel(m=m_, M=M_, C=C_, *args, **kwargs)

#| Define CMB sampling class

from scipy.stats import chi2

class CMB(object):
    def __init__(self, Cl):
        self.Cl = Cl

    def rvs(self, shape=()):
        shape = tuple(np.atleast_1d(shape))
        return chi2(2*l+1).rvs(shape + self.Cl.shape)*self.Cl/(2*l+1)

    def logpdf(self, x):
        return (chi2(2*l+1).logpdf((2*l+1)*x/self.Cl)  + np.log(2*l+1)-np.log(self.Cl)).sum(axis=-1) 

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX 
emulator = CosmoPowerJAX(probe='cmb_tt')
paramnames = [('Ωbh2', r'\Omega_b h^2'), ('Ωch2', r'\Omega_c h^2'), ('h', 'h'), ('τ', r'\tau'), ('ns', r'n_s'), ('lnA', r'\ln(10^{10}A_s)')]
params = ['Ωbh2', 'Ωch2', 'h', 'τ', 'ns', 'lnA']
θmin, θmax = np.array([[0.01865, 0.02625], [0.05, 0.255], [0.64, 0.82], [0.04, 0.12], [0.84, 1.1], [1.61, 3.91]]).T
l = np.arange(2, 2509)

#| Define the observed variables, set seed for observed, random seed for the analysis
np.random.seed(0)
θobs = np.array([0.02225,0.120,0.693,0.054,0.965,3.05])
Dobs = CMB(emulator.predict(θobs)).rvs()
np.savetxt("theta.csv", θobs)
np.savetxt("data.csv", Dobs)
np.random.seed()



#| If you want to reproduce the ground-truth yourself, uncomment and run the below (takes about an hour on four cores)

#from pypolychord import run
#samples = run(lambda θ: CMB(emulator.predict(θ)).logpdf(Dobs), len(θmin), prior=lambda x: θmin + (θmax-θmin)*x, paramnames=paramnames)
#samples.to_csv('lcdm.csv')

#| Otherwise just load these chains

from anesthetic import read_chains
import os
jaxsamples = read_chains(os.path.join(os.path.dirname(__file__), 'jaxLCDM.csv'))

#| Wrap cosmopowerjax predictions with this to check that only physical simulations are generated
def Generate_Cl(Nsim,model,i):
    θ_ = model.rvs(Nsim)
    predictions = emulator.predict(θ_)
    θ_ = θ_[~np.isinf(predictions).any(axis=1)]
    predictions = predictions[~np.isinf(predictions).any(axis=1)]
    breakcondition = 0
    if len(predictions) < Nsim//2:
        print(f"Bad Posterior on iteration {i+1}")
        raise ValueError("Bad Posterior")
    while len(predictions) < Nsim and breakcondition < 10:
        θ_ = np.concatenate([θ_,model.rvs(Nsim-len(predictions))])
        predictions = emulator.predict(θ_)
        θ_ = θ_[~np.isinf(predictions).any(axis=1)]
        predictions = predictions[~np.isinf(predictions).any(axis=1)]
        breakcondition += 1
    if breakcondition == 10:
        print(f"Too many unphysical simulations in iteration {i}, reverting to previous model")
        raise ValueError("Bad Posterior")
    return θ_, predictions

#| Run sequential LSBI
def run_LSBI(θ, D, Dobs, n_runs=4):
    models =[]
    i = 0
    for i in tqdm.trange(n_runs):
        generated = False
        while not generated:
            if i == 0:
                models = [LSBI(θ, D, μ= (θmin + θmax)/2, Σ= ((θmax - θmin)/2)**2, shape=N_shape)]
            else:
                models.append(LSBI(θ_, D_, μ=models[-1].μ, Σ=models[-1].Σ, shape=N_shape))
            if i < n_runs-1:
                try:
                    currmodel = models[-1].posterior(Dobs)
                    θ_, Cl_ = Generate_Cl(Nsim,currmodel,i)
                    D_ = CMB(Cl_).rvs()
                    generated = True
                except Exception as e:
                    models.pop()
                    print(f"Error occurred: {e}. Retrying iteration {i+1}.")
            else:
                generated = True
    return models

import tqdm
## Create initial simulations
n_params = emulator.n_parameters
θ = np.random.normal(loc=(θmin + θmax) / 2, scale=(θmax - θmin) / 6, size=(Nsim, n_params))
Cl = emulator.predict(θ)
D = CMB(Cl).rvs()
models=(run_LSBI(θ,D,Dobs,n_runs))

#| Plot the results

from anesthetic.plot import make_2d_axes
fig,axes = make_2d_axes(params, labels=jaxsamples.get_labels_map(), figsize=(7,7))

#| Set Plotting Limits, 6 sigma away from the centre of one of the normals in the mixturenormal (CHANGE THIS)
finalpost = models[-1].posterior(Dobs)
finalstd = np.sqrt(np.diag(finalpost.cov[0]))
lowerlim = θobs- 6*finalstd
upperlim = θobs + 6*finalstd
for i, p in enumerate(params):
    axes.loc[p, p].set_xlim(lowerlim[i], upperlim[i])

from lsbi.stats import multivariate_normal, mixture_normal
for n in range(n_runs+1):
    if n == 0:
        posterior = multivariate_normal(mean=(θmin + θmax)/2, cov=((θmax - θmin)/2)**2) ## Prior
        posterior.plot_2d(axes,label='Prior', color='black', alpha=0.2)
    else:
        posterior = models[n-1].posterior(Dobs)
        means = posterior.mean
        covs = posterior.cov
        postcopy = mixture_normal(mean=means, cov=covs)
        postcopy.plot_2d(axes,label=f'run {n}', color=f'C{n-1}', alpha=0.2+(0.6/n_runs)*n)

axes.iloc[-1, 0].legend(loc='lower center', bbox_to_anchor=(len(axes)/2, len(axes)), ncol=6)
axes.axlines(dict(zip(params, θobs)), color='k', ls='--')

fig.savefig(filename, format="pdf", bbox_inches='tight')
