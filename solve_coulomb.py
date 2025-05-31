from varpro import *
x=np.logspace(-2,3,500)
y=1/x
weights=x
from copy import deepcopy
#Step 1: Fit one Gaussian
num_real= 3
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
for i in range(15):
    new = deepcopy(best_params)
    #print(new)
    #Approach 1: 2 real Gaussians
    avals=np.concatenate((new["realGaussians_unshifted"], np.logspace(-2,2, 2))) #Outlier Gaussians
    new= {
        "realGaussians_unshifted": (avals,),
    }
    #print(new)
    function_types_and_amounts = [(ftype,len(v[0])) for ftype, v in new.items()]

    varpro = VarPro(function_types_and_amounts, x, y, weights)
    best_params_oneGaussian,error=varpro.fit(new)
    error_one_gaussian = error

    #Approach 2: 1 pair of complex Gaussians
    avals=best_params["realGaussians_unshifted"]
    avals_twoGaussians=[1]
    bvals=[1]
    new= {
        "realGaussians_unshifted": (avals,),
        "complexGaussians_unshifted_pairs": (avals_twoGaussians, bvals),
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
"""
approximation=varpro.eval_function(best_params,x)
import matplotlib.pyplot as plt
plt.plot(x, y, label='Original Data')
plt.plot(x, approximation, label='Fitted Function')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Coulomb Fitting')
plt.show()
params_dict = {
    "realGaussians_unshifted": (np.logspace(-2,2, num_real),),
    "complexGaussians_unshifted_pairs": (np.logspace(-1,1,num_complex), np.linspace(2,3,num_complex)),
}
"""