import os
import warnings

from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
import numpy as np

from methods import romberg, golden_section


def number_density_profile(x, a, b, c, A, N,):
    """Number density profile"""
    return 4*np.pi*x**2*A*N*(x/b)**(a-3)*np.exp(-(x/b)**c)

def binned_number_density_profile(binedges, a, b, c, A, N):
    """Number density profile integrated over bins"""
    N = romberg(
        f=lambda x: 4*np.pi*x**2*A*N*(x/b)**(a-3)*np.exp(-(x/b)**c),
        start=binedges[0], 
        end=binedges[1], 
        m=10
    )
    return N

def dnda(x, a, b, c, A, N):
    """Derivative of number density profile with respect to a"""
    n = number_density_profile(x, a, b, c, A, N)
    return n * np.log(x/b)

def dndb(x, a, b, c, A, N):
    """Derivative of number density profile with respect to b"""
    n = number_density_profile(x, a, b, c, A, N)
    return n * -(a-3-c*(x/b)**c)/b

def dndc(x, a, b, c, A, N):
    """Derivative of number density profile with respect to c"""
    n = number_density_profile(x, a, b, c, A, N)
    return n * np.log(x/b) * (x/b)**c

def calculate_A(a, b, c, N):
    """Calculate normalisation factor of number density profile"""
    f = lambda x: (x/b)**(a-1)*np.exp(-(x/b)**c)
    integral = romberg(f, 0, 5, 10)
    A = 1/(4*np.pi*b**2*integral)
    return A

def chi2(binedges, datay, model, a, b, c):
    """Calculate chi^2 value of a number density model given datay/

    Args:
        binedges (list): List of length len(datay)+1 containing edges of the 
            used bins to generate data.
        datay (list): Data in bins binedges.
        model (calable): function to calculate model predictions. Function 
            should have atributes: binedges(tuple), a(float), b(float), 
            c(float), A(float).
        a (flaot): number density profile parameter a
        b (flaot): number density profile parameter b
        c (flaot): number density profile parameter c

    Returns:
        float: chi^2 value
    """
    # Calculate normalization factor
    A = Anorm(a,b,c)
    # integrate model over bins
    modely = [
        model(
            binedges[x:x+2], a, b, c, A
        ) for x in range(len(binedges)-1)
    ]
    return sum( (datay - modely)**2 / modely )

def grad_chi2(binedges, datay, model, model_deriv, Anorm, params):
    """Calculate the gradient of chi^2 of a number density profile given datay.

    Args:
        binedges (list): List of length len(datay)+1 containing edges of the 
            used bins to generate data.
        datay (list): Data in bins binedges.
        model (calable): function to calculate model predictions. Function 
            should have atributes: binedges(tuple), a(float), b(float), 
            c(float), A(float).
        model_deriv (list,calable): List of function to calculate model 
            derivatives. Functions should have atributes: binedges(tuple), 
            a(float), b(float), c(float), A(float).
        Anorm (callable): Function to calculate A. Functions should have 
            atributes: a(float), b(float), c(float), A(float).
        params (list): list of float containig the parameter values a, b and c.

    Returns:
        np.array: list containig gradient of chi^2
    """
    
    model_deriv_at_params = []
    
    # Calculate model derivatives at params
    for dmdk in model_deriv:
        model_deriv_at_params.append(np.array([
            romberg(
                f=lambda x: dmdk(
                    x, 
                    a=params[0],
                    b=params[1], 
                    c=params[2],
                    A=Anorm
                ),
                start=binedges[i], 
                end=binedges[i+1], 
                m=10
            ) for i in range(len(binedges)-1)
        ]))
    model_deriv_at_params = np.array(model_deriv_at_params)
    
    # Calculate model parameters at params
    modely = np.array([
        model(
            binedges[x:x+2], *params, Anorm
        ) for x in range(len(binedges)-1)
    ])

    grad = np.zeros(np.shape(model_deriv))
    for idx, dmdk in enumerate(model_deriv_at_params):
        # Calculate derivative of chi^2 with respect to k
        insum = (datay**2/modely**2 - 1) * dmdk
        # ensure no nan values
        insum[np.isneginf(insum)]=0
        grad[idx] = sum(insum)
    return grad

def log_likelihood_poisson(binedges, datay, model, a, b, c):
    """Calculate minus log likelihood value of a number density model given datay.

    Args:
        binedges (list): List of length len(datay)+1 containing edges of the 
            used bins to generate data.
        datay (list): Data in bins binedges.
        model (calable): function to calculate model predictions. Function 
            should have atributes: binedges(tuple), a(float), b(float), 
            c(float), A(float).
        a (flaot): number density profile parameter a
        b (flaot): number density profile parameter b
        c (flaot): number density profile parameter c

    Returns:
        float: -ln(L)
    """
    # calculate normalization A
    A = Anorm(a,b,c)
    # Integrate model over bins
    modely = [
        model(
            binedges[x:x+2], a, b, c, A
        ) for x in range(len(binedges)-1)
    ]
    # calculate -ln(L)
    return -sum( datay*np.log(modely) - modely)

def grad_log_likelihood_poisson(binedges, datay, model, model_deriv, Anorm, params):
    """Calculate the gradient of minus log likelihoof of a number density 
    profile given datay.

    Args:
        binedges (list): List of length len(datay)+1 containing edges of the 
            used bins to generate data.
        datay (list): Data in bins binedges.
        model (calable): function to calculate model predictions. Function 
            should have atributes: binedges(tuple), a(float), b(float), 
            c(float), A(float).
        model_deriv (list,calable): List of function to calculate model 
            derivatives. Functions should have atributes: binedges(tuple), 
            a(float), b(float), c(float), A(float).
        Anorm (callable): Function to calculate A. Functions should have 
            atributes: a(float), b(float), c(float), A(float).
        params (list): list of float containig the parameter values a, b and c.

    Returns:
        np.array: list containig gradient of -ln(L)
    """
    model_deriv_at_params = []

    # Calculate model derivatives at params
    for dmdk in model_deriv:
        model_deriv_at_params.append(np.array([
            romberg(
                f=lambda x: dmdk(
                    x, 
                    a=params[0],
                    b=params[1], 
                    c=params[2],
                    A=Anorm
                ),
                start=binedges[i], 
                end=binedges[i+1], 
                m=10
            ) for i in range(len(binedges)-1)
        ]))
    model_deriv_at_params = np.array(model_deriv_at_params)
    
    # Calculate model parameters at params
    modely = np.array([
            model(
                binedges[x:x+2], *params, Anorm
            ) for x in range(len(binedges)-1)
        ])

    grad = np.zeros(np.shape(model_deriv))
    for idx, dmdk in enumerate(model_deriv_at_params):
        # Calculate derivative of -ln(L) with respect to k
        insum = (datay/modely - 1) * dmdk
        # ensure no nan values
        insum[np.isneginf(insum)]=0
        grad[idx] = sum(insum)
    return grad

def quasi_newton_chi2(
        binedges, datay, model, model_deriv, Anorm, start, maxit=100, 
        target_acc=1e-6, verbose=True
    ):
    """Quasi-Newton Algorithm for optimizing a chi^2 estimator for a number 
    denisity profile. Function makes use of BFGS methode for updating hessian.

    Args:
        binedges (list): List of length len(datay)+1 containing edges of the 
            used bins to generate data.
        datay (list): Data in bins binedges.
        model (calable): function to calculate model predictions. Function 
            should have atributes: binedges(tuple), a(float), b(float), 
            c(float), A(float).
        model_deriv (list,calable): List of function to calculate model 
            derivatives. Functions should have atributes: binedges(tuple), 
            a(float), b(float), c(float), A(float).
        Anorm (callable): Function to calculate A. Functions should have 
            atributes: a(float), b(float), c(float), A(float).
        start (list): starting position of a, b, c.
        maxit (int, optional): Maximum number of iterations. Defaults to 100.
        target_acc (float, optional): Target accuracy. Defaults to 1e-6.
        verbose (bool, optional): Print warning and progress. Defaults to True.

    Returns:
        tuple(list,float): final a, b, c and chi^2
    """
    xi = np.array(start)
    # initial hessian
    Hi = np.diag([1 for _ in range(len(xi))])
    i = 0
    while i < maxit:
        
        # impose limit on a
        if xi[0] <0:
            xi[0] = start[0]
        # impose limit on b
        if xi[1] <0:
            xi[1] = start[1]
        # impose limit on c
        if xi[2] <0:
            xi[2] = start[2]

        # Calculate A(a,b,c)
        Anorm_i = Anorm(*xi)

        # Calculate gradiant at (a,b,c)
        gradf_i = grad_chi2(
            binedges=binedges,
            datay=datay,
            model=model,
            model_deriv=model_deriv,
            Anorm=Anorm_i,
            params=xi)
        
        # obtain direction for the next step
        ni = -Hi.dot(gradf_i)
        # minimize x_i+1 = x_i + l_i*n_i using golden section algorithm
        func_l = lambda l: chi2(
            binedges, datay, model, *[x+l*n for x, n in zip(xi,ni)]
        )
        l = golden_section(func_l, 0, 1e-6, target_acc=1e-6, maxit=100)

        x_i1 = xi + l*ni

        # Check if nans in golden section
        if any(np.isnan(x) for x in x_i1):
            if verbose:
                print('Nan encoutered in golden section.')
            x_i1 = xi + 1e-9*ni


        delta_i1 = l*ni
        
        D_i1 = \
            grad_chi2(
                binedges=binedges,
                datay=datay,
                model=model,
                model_deriv=model_deriv,
                Anorm=Anorm_i,
                params=x_i1
            ) - gradf_i

        # Check for convergence
        if all(abs(d)<target_acc for d in D_i1):
            if verbose:
                print('target acc reached.')
            break
        
        # Update hessian according to BFGS method
        HD = Hi.dot(D_i1)
        
        u = delta_i1/(delta_i1 @ D_i1) - (HD)/(D_i1 @ (HD))
        
        Hi = Hi \
               + np.outer(delta_i1, delta_i1)/(delta_i1 @ D_i1) \
               - np.outer(HD, HD)/(D_i1 @ HD) \
               + (D_i1 @ HD)*np.outer(u,u)
        
        # take the step
        xi = x_i1
        i += 1
    # Calculate final chi2
    chi2_i = chi2(binedges, datay, model, *xi) 
    return xi, chi2_i

def quasi_newton_poisson(
        binedges, datay, model, model_deriv, Anorm, start, maxit=100, 
        target_acc=1e-6, verbose=True
    ):
    """Quasi-Newton Algorithm for optimizing a minus log likelihood of a Poisson
    estimator for a number denisity profile. Function makes use of BFGS methode 
    for updating hessian.

    Args:
        binedges (list): List of length len(datay)+1 containing edges of the 
            used bins to generate data.
        datay (list): Data in bins binedges.
        model (calable): function to calculate model predictions. Function 
            should have atributes: binedges(tuple), a(float), b(float), 
            c(float), A(float).
        model_deriv (list,calable): List of function to calculate model 
            derivatives. Functions should have atributes: binedges(tuple), 
            a(float), b(float), c(float), A(float).
        Anorm (callable): Function to calculate A. Functions should have 
            atributes: a(float), b(float), c(float), A(float).
        start (list): starting position of a, b, c.
        maxit (int, optional): Maximum number of iterations. Defaults to 100.
        target_acc (float, optional): Target accuracy. Defaults to 1e-6.
        verbose (bool, optional): Print warning and progress. Defaults to True.

    Returns:
        tuple(list,float): final a, b, c and -ln(L)
    """
    xi = np.array(start)
    # initial hessian
    Hi = np.diag([1 for _ in range(len(xi))])
    i = 0
    while i < maxit:
        
        # impose limit on a
        if xi[0] <0:
            xi[0] = start[0]
        # impose limit on b
        if xi[1] <0:
            xi[1] = start[1]
        # impose limit on c
        if xi[2] <0:
            xi[2] = start[2]

        # Calculate A(a,b,c)
        Anorm_i = Anorm(*xi)

        # Calculate gradiant at (a,b,c)
        gradf_i = grad_log_likelihood_poisson(
            binedges=binedges,
            datay=datay,
            model=model,
            model_deriv=model_deriv,
            Anorm=Anorm_i,
            params=xi
        )

        # obtain direction for the next step
        ni = -Hi.dot(gradf_i)
        # minimize x_i+1 = x_i + l_i*n_i using golden section algorithm
        func_l = lambda l: log_likelihood_poisson(
            binedges, datay, model, *[x+l*n for x, n in zip(xi,ni)]
        )
        
        l = golden_section(func_l, 0, 1e-5, target_acc=1e-5, maxit=100)
        
        x_i1 = xi + l*ni

        # Check if nans in golden section
        if any(np.isnan(x) for x in x_i1):
            if verbose:
                print('Nan encoutered in golden section.')
            x_i1 = xi + 1e-9*ni

        delta_i1 = l*ni
        
        D_i1 = \
            grad_log_likelihood_poisson(
                binedges=binedges,
                datay=datay,
                model=model,
                model_deriv=model_deriv,
                Anorm=Anorm_i,
                params=x_i1
            ) - gradf_i

        # Check for convergence
        if all(abs(d)<target_acc for d in D_i1):
            if verbose:
                print('target acc reached.')
            break
        
        # Update hessian according to BFGS method
        HD = Hi.dot(D_i1)
        
        u = delta_i1/(delta_i1 @ D_i1) - (HD)/(D_i1 @ (HD))
        
        Hi = Hi \
               + np.outer(delta_i1, delta_i1)/(delta_i1 @ D_i1) \
               - np.outer(HD, HD)/(D_i1 @ HD) \
               + (D_i1 @ HD)*np.outer(u,u)
        
        # take the step
        xi = x_i1
        i += 1
    # Calculate final -ln(L)
    log_likelihood_poisson_i = log_likelihood_poisson(
        binedges, datay, model, *xi
    )
    return xi, log_likelihood_poisson_i

def G_test(expected, observed):
    """G test given a set of expected values and observed values

    Args:
        expected (list): list of expected values
        observed (list): list of observed values

    Returns:
        float: Calculated G value
    """
    G=0
    for exp, obs in zip(expected, observed):
        # Since lim x->0 xlog(1/x) = 0 empty observed bins are skiped
        if obs != 0:
            G+= obs*np.log(exp/obs)
    G *= 2

    return G

def chi2distribution(x,k):
    """chi^2 distribution for statistic value x and degrees of freedom k

    Args:
        x (float): obtained statistic
        k (int): degrees of freedom

    Returns:
        float: chi^2 distribution value at (x,k) 
    """
    return gammainc(k/2, x/2) / gamma(k/2)

def signigficance_of_fit(expected, observed, k):
    """Calculate the significance of a model given a set observations assuming 
    Chi^2 distributions function of statistic.

    Args:
        expected (list): list of expected values
        observed (list): list of observed values
        k (int): degrees of freedom of the model

    Returns:
        float: Calculated significance
    """
    G = G_test(expected, observed)
    P = chi2distribution(G, k)
    return 1 - P

if __name__=='__main__':
    
    # Constants
    DATA_DIR = './data/'
    PLOT_DIR = './plots/'
    OUTPUT_DIR =  './output/'
    PLOT = True

    # Loading the data
    data = [
        np.loadtxt(
            os.path.join(DATA_DIR, 'satgals_m{}.txt'.format(i)), 
            skiprows=4
        ) for i in range(11,15+1,1)
    ]

    Nhalos = [
        np.loadtxt(
            os.path.join(DATA_DIR, 'satgals_m{}.txt'.format(i)), 
            max_rows=1
        ) for i in range(11,15+1,1)
    ]

    # Calculating <N_sat>
    Nsat = [len(d)/int(n) for d,n in zip(data, Nhalos)]

    # Binning the data
    binedges = np.logspace(-4, np.log10(5), 15+1) # 25 bins

    binned_mean_sat_per_halo = []
    for d, nh_i in zip(data, Nhalos):
        hist, _ = np.histogram(d[:,0], binedges)
        binned_mean_sat_per_halo.append(hist/nh_i)
    
    # 1a

    # Set initial guess for a, b, c
    initial_guess = [2.4, 0.25, 1.6]

    fitted_params_chi2 = []
    final_chi2 = []
    for idx, (datay_i, Nsat_i) in enumerate(zip(binned_mean_sat_per_halo, Nsat)):
        
        # Define model for given <Nsat>
        model = lambda x,a,b,c,A: binned_number_density_profile(x,a,b,c,A,Nsat_i)
        # Define model derivatives for given <Nsat>
        model_deriv = [
            lambda x,a,b,c,A: dnda(x,a,b,c,A,Nsat_i),
            lambda x,a,b,c,A: dndb(x,a,b,c,A,Nsat_i),
            lambda x,a,b,c,A: dndc(x,a,b,c,A,Nsat_i)
        ]
        # Normalization calculator function for given <Nsat>
        Anorm = lambda a,b,c: calculate_A(a,b,c,Nsat_i)
    
        # performing fit
        fit, chi2i = quasi_newton_chi2(
            binedges = binedges,
            datay = datay_i, 
            model = model, 
            model_deriv = model_deriv, 
            Anorm = Anorm, 
            start = initial_guess, 
            maxit = 25, 
            target_acc = 1e-6, 
            verbose = False
        )
        fitted_params_chi2.append(fit)
        final_chi2.append(chi2i)
        
        # Write results to file
        with open(
                os.path.join(
                    OUTPUT_DIR, 'chi2_satgal_m{}_best_fit_a.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write('{:.3f}'.format(fit[0]))
        with open(
                os.path.join(
                    OUTPUT_DIR, 'chi2_satgal_m{}_best_fit_b.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write('{:.3f}'.format(fit[1]))
        with open(
                os.path.join(
                    OUTPUT_DIR, 'chi2_satgal_m{}_best_fit_c.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write('{:.3f}'.format(fit[2]))
        with open(
                os.path.join(
                    OUTPUT_DIR, 'chi2_satgal_m{}.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write('{:.5f}'.format(chi2i))
    
    # 1b
    fitted_params_poisson = []
    final_likelihood = []
    for idx, (datay_i, Nsat_i) in enumerate(zip(binned_mean_sat_per_halo, Nsat)):
        
        # Define model for given <Nsat>
        model = lambda x,a,b,c,A: binned_number_density_profile(x,a,b,c,A,Nsat_i)
        # Define model derivatives for given <Nsat>
        model_deriv = [
            lambda x,a,b,c,A: dnda(x,a,b,c,A,Nsat_i),
            lambda x,a,b,c,A: dndb(x,a,b,c,A,Nsat_i),
            lambda x,a,b,c,A: dndc(x,a,b,c,A,Nsat_i)
        ]
        # Normalization calculator function for given <Nsat>
        Anorm = lambda a,b,c: calculate_A(a,b,c,Nsat_i)

        # performing fit
        fit, likelihood = quasi_newton_poisson(
            binedges = binedges, 
            datay = datay_i, 
            model = model, 
            model_deriv = model_deriv, 
            Anorm = Anorm, 
            start = initial_guess, 
            maxit = 25, 
            target_acc = 1e-6, 
            verbose = False
        )
        fitted_params_poisson.append(fit)
        final_likelihood.append(likelihood)
        
        # plotting results of chi^2 fit and Poisson fit in the same figure.
        if PLOT:
            plt.figure(figsize=(5,4))
            plt.loglog(
                binedges[:-1], 
                [
                    model(
                        binedges[x:x+2], *fitted_params_chi2[idx], Anorm(*fitted_params_chi2[idx])
                    ) for x in range(len(binedges)-1)
                ], 
                label='fit r$\chi^2$',
                color ='C0',
                ls='--'
            )
            plt.loglog(
                binedges[:-1], 
                [
                    model(
                        binedges[x:x+2], *fit, Anorm(*fit)
                    ) for x in range(len(binedges)-1)
                ], 
                label='fit Poisson',
                color='C0',
                ls='-'
            )
            plt.errorbar(binedges[:-1], datay_i, yerr=datay_i, ls='', marker='o', label='data', color='C1')
            plt.title(r'$\log(M_\odot/h) = $'+'{}'.format(idx+11))
            plt.axis(xmin=binedges[0], xmax=binedges[-1])
            plt.xlabel(r'$x$')
            plt.ylabel(r'$N(x)$')

            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, '1_{}.png'.format(idx)))
            plt.clf()

        # Write results to file
        with open(
                os.path.join(
                    OUTPUT_DIR, 'lnL_satgal_m{}_best_fit_a.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write('{:.3f}'.format(fit[0]))
        with open(
                os.path.join(
                    OUTPUT_DIR, 'lnL_satgal_m{}_best_fit_b.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write('{:.3f}'.format(fit[1]))
        with open(
                os.path.join(
                    OUTPUT_DIR, 'lnL_satgal_m{}_best_fit_c.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write('{:.3f}'.format(fit[2]))
        with open(
                os.path.join(
                    OUTPUT_DIR, 'lnL_satgal_m{}.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write('{:.3f}'.format(likelihood))
    

    #1c
    for idx, datay_i in enumerate(binned_mean_sat_per_halo):
        N = len(datay_i) -1            # data points
        M = 3                          # parameters
        k = N-M                        # degrees of freedom

        # Calculate expected values
        modely_i = number_density_profile(
            binedges[:-1],
            *fitted_params_chi2[idx],
            calculate_A(
                *fitted_params_chi2[idx],
                Nsat[idx]
            ),
            Nsat[idx]
        )
        # Calculate significance
        chi2_significance = signigficance_of_fit(
            expected=modely_i[datay_i>0], 
            observed=datay_i[datay_i>0], 
            k=k
        )

        # Calculate Expected values
        modely_i = number_density_profile(
            binedges[:-1],
            *fitted_params_poisson[idx],
            calculate_A(
                *fitted_params_poisson[idx],
                Nsat[idx]
            ),
            Nsat[idx]
        )
        # Calculate significance
        poisson_significance = signigficance_of_fit(
            expected=modely_i[datay_i>0], 
            observed=datay_i[datay_i>0], 
            k=k
        )

        # Write results to file
        with open(
                os.path.join(
                    OUTPUT_DIR, 
                    'chi2_satgal_m{}_significance_of_fit.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write(str(chi2_significance))
        
        with open(
                os.path.join(
                    OUTPUT_DIR, 
                    'lnL_satgal_m{}_significance_of_fit.txt'.format(idx+11)
                ), 'w'
            ) as f:
            f.write(str(poisson_significance))
