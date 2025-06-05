import numpy as np
from numpy import sinh, cosh, pi, sqrt
import matplotlib.pyplot as plt
import sys
import scipy
from scipy.special import erf
#Sinc quadrature according to https://www.sciencedirect.com/science/article/pii/S0377042700003484
def sinc_weights_exponents(width: float, n_gauss: int,scaler: float = 1.0):
    """
    Compute Sinc–Gauss weights w_k and exponents e_k for the approximation

        1/r ≈ ∑_k w_k exp(-e_k r**2)
    """
    k = np.arange(n_gauss)                 
    h = 2 * width / (2 * n_gauss - 1)      
    t = -width + k * h                     

    w = np.cosh(t)
    w[1:] *= 2                            
    w *= h / np.sqrt(np.pi)              

    e = np.sinh(t) ** 2                 
    return w*scaler, e*scaler**2


def inv_r(r, w, e):
    """Vectorised 1/r approximant."""
    r = np.atleast_1d(r)
    return np.dot(w, np.exp(-np.outer(e, r**2)))


# --- Parameters identical to the original script ------------------------------
def inv(rvals,weights,exponents):
    result=0
    counter=0
    for i in range(len(weights)):
            result+= weights[i]*np.exp(-exponents[i]*rvals**2)
    return result
if __name__ == "__main__":
    b= 12
    N = 90
    alpha = 5e-4
    B_WIDTH = b
    N_GAUSS = N
    SCALER  = alpha

    weights, exponents = sinc_weights_exponents(B_WIDTH, N_GAUSS,scaler=SCALER)
    mu=sqrt(pi)/2*inv(1e-8,weights,exponents)

    r=np.logspace(-5,5,10000)
    #r=np.linspace(1e-5,10,int(1e6))
    #plt.plot(r,r*inv(r,weights,exponents),label="approx")
    #exponents=exponents*scaler**2
    #weights=weights*scaler
    

    fig, axs = plt.subplots(2, 1, figsize=(8, 7))
    axs[0].plot(r,abs((inv(r,weights,exponents)-1/r)*r),label="1/r")
    axs[0].plot(r,abs((inv(r,weights,exponents)-erf(mu*r)/r)*r),label="erfmu")
    axs[0].legend()
    axs[0].set_ylim(1e-16,2)

    print(inv(1e-4,weights,exponents))
    print("mu=%f"%mu)

    plt.suptitle(r"Sinc quadrature of 1/r with $K=%d$ points."%(len(weights)))
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("r")
    axs[0].set_ylabel("Relative error ")
    axs[1].plot(r,inv(r,weights,exponents),label="approximate 1/r")
    axs[1].plot(r,1/r,label="exact 1/r")
    axs[1].plot(r,erf(mu*r)/r,label="erfmu")

    axs[1].legend()
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_ylim(1e-10,1e10)
    plt.tight_layout()
    plt.savefig("Sinc_oneoverr_1e-1.png")

    plt.show()
    print("exponents")
    print(list(exponents))
    print("weights")
    print(list(weights))
