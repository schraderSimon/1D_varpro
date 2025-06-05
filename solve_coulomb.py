from varpro import *
from scipy.special import erf
import math
x = np.logspace(-3, 3, 1000).astype(ddouble)
def erf_over_r(mu, r):
    # math.erf needs a Python float; cast back to ddouble afterwards
    return ddouble.type(math.erf(float(mu * r)) / float(r)) if r != 0 else (
            ddouble.type(2) * mu / ddouble.type(math.sqrt(math.pi)))

mu=40
y=np.array([erf_over_r(mu, r) for r in x], dtype=ddouble)
weights=1/y
from copy import deepcopy
#Step 1: Fit three Gaussians
num_real= 1
num_complex= 0



params_dict = {
    "realGaussians_unshifted": (np.logspace(-2,2, num_real),),
}
print(params_dict)
function_types_and_amounts = [(ftype,len(v[0])) for ftype, v in params_dict.items()]
varpro = VarPro(function_types_and_amounts, x, y, weights)
best_params,error=varpro.fit(params_dict)
print(error)
print(best_params)

#Step 2: Fit two Gaussians
for i in range(20):
    new = deepcopy(best_params)
    #print(new)
    #Approach 1: 2 real Gaussians
    avals=np.concatenate((new["realGaussians_unshifted"][0], np.logspace(-2,2, 2))) #Outlier Gaussians
    try:
        avalsC,bvalsC=new["complexGaussians_unshifted_pairs"]
        new= {
            "realGaussians_unshifted": (avals,) ,
            "complexGaussians_unshifted_pairs":(avalsC,bvalsC)
        }
        len_avals_c=len(avalsC)
    except KeyError:
        pass
        new= {
            "realGaussians_unshifted": (avals,),
        }
        len_avals_c=0
    print("Number of Gaussians in total: %d"%(len(avals)+len_avals_c*2))
    function_types_and_amounts = [(ftype,len(v[0])) for ftype, v in new.items()]

    varpro = VarPro(function_types_and_amounts, x, y, weights)
    best_params_oneGaussian,error=varpro.fit(new)
    error_one_gaussian = error
    #Approach 2: 1 pair of complex Gaussians
    avals=best_params["realGaussians_unshifted"][0]
    try:
        avalsC,bvalsC=new["complexGaussians_unshifted_pairs"]
        new= {
           "realGaussians_unshifted": (avals,) ,
            "complexGaussians_unshifted_pairs":(np.concatenate((avalsC,[1])),np.concatenate((bvalsC,[1])))
        }
        len_avals_c=len(avalsC)
    except KeyError:
        pass
        new= {
                "realGaussians_unshifted": (avals,),
                "complexGaussians_unshifted_pairs": ([1], [1]),
            }
    function_types_and_amounts = [(ftype,len(v[0])) for ftype, v in new.items()]
    varpro = VarPro(function_types_and_amounts, x, y, weights)
    best_params_twoComplex,error=varpro.fit(new)
    error_two_complex = error
    print(f"Iteration {i+1}: Error with one Gaussian: {error_one_gaussian}, Error with two complex Gaussians: {error_two_complex}")
    if error_one_gaussian < error_two_complex:
        best_params = best_params_oneGaussian
        print("Using two different real Gaussians")
    else:
        best_params = best_params_twoComplex
        print("Using complex Gaussians")
    new=best_params
    avals=new["realGaussians_unshifted"][0] #Outlier Gaussians
    try:
        avalsC,bvalsC=new["complexGaussians_unshifted_pairs"]
        
        new= {
                "realGaussians_unshifted": (avals,),
                "complexGaussians_unshifted_pairs":(avalsC,bvalsC)
            }
    except KeyError:
        new= {
            "realGaussians_unshifted": (avals,),
        }
    function_types_and_amounts = [(ftype,len(v[0])) for ftype, v in new.items()]
    print(function_types_and_amounts)
    varpro = VarPro(function_types_and_amounts, x, y, weights)
    varpro.fit(new)
    approximation=varpro.eval_function(new,x)
    biggest_relative_difference=max (abs(y-approximation)/y)
    print("Biggest relative error: %e, biggest complex: %e"%(biggest_relative_difference,max(np.imag(approximation))))
import matplotlib.pyplot as plt
error = np.abs(y - approximation)/ np.abs(y)

fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Top subplot: original vs. fitted
axs[0].plot(x, y, label='Original Data')
axs[0].plot(x, approximation, label='Fitted Function')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].set_title('Coulomb Fitting')

# Bottom subplot: error
axs[1].plot(x, error, label='Relative Error')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Error')
axs[1].legend()
axs[1].set_ylim(1e-16,1)
axs[0].set_ylim(1e-16,1)
plt.tight_layout()
plt.show()