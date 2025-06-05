import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import minimize
import sys
from xprec import ddouble      


data_type=ddouble  # Use ddouble for high precision calculations
import xprec.linalg as xl              
from abc import ABC, abstractmethod
np.set_printoptions(linewidth=300)
_arrays_per_type = {
            "realGaussians_unshifted":          1,
            "complexGaussians_unshifted":       2,
            "complexGaussians_unshifted_pairs": 2,
        }
_function_type_metadata = {
    "realGaussians_unshifted": {
        "num_params": 1,
        "num_functions": 1,
    },
    "complexGaussians_unshifted": {
        "num_params": 2,
        "num_functions": 1,
    },
    "complexGaussians_unshifted_pairs": {
        "num_params": 2,
        "num_functions": 2,
    },
}
_factory = {
    "realGaussians_unshifted":          lambda p: realGaussians_unshifted(p),
    "complexGaussians_unshifted":       lambda p: complexGaussians_unshifted(*p),
    "complexGaussians_unshifted_pairs": lambda p: complexGaussians_unshifted_pairs(*p),
}
_function_type_metadata["complexshiftedGaussians"] = {
    "num_params": 4,
    "num_functions": 1,
}

_factory["complexshiftedGaussians"] = (
    lambda p: complexshiftedGaussians(*p)
)
def dX_column_for_param(param_idx, col_idx,
                        function_types_and_amounts,
                        derivs, x_weights=None):
    """
    Return dX[:, col_idx] / dα_param_idx    as a 1-D array of length N.

    Parameters
    ----------
    param_idx : int
        Flat index of the nonlinear parameter α_i.
    col_idx   : int
        Column k in the X matrix for which the derivative is requested.
    function_types_and_amounts : list[(str,int)]
        Same structure you already pass to VarPro.
    derivs : list[np.ndarray]
        Output of the loop you sketched – one tensor per block.
    x_weights : 1-D array or None
        If you need the weighted derivative, pass self.weights,
        otherwise keep it None.
    """
    p_cur = 0        # cursor in the parameter vector
    c_cur = 0        # cursor in the column space of X

    for (ftype, amount), d in zip(function_types_and_amounts, derivs):
        meta      = _function_type_metadata[ftype]
        n_params  = meta["num_params"]
        n_funcs   = meta["num_functions"]

        p_block   = amount * n_params
        c_block   = amount * n_funcs

        # Does α_param_idx live in this block?
        if p_cur <= param_idx < p_cur + p_block:
            local      = param_idx - p_cur        # 0‥p_block-1
            param_kind = local // amount          # which parameter in the tuple
            basis_idx  = local %  amount          # which basis element

            if n_funcs == 1:                      # real or single complex Gaussian
                if col_idx != c_cur + basis_idx:
                    raise ValueError("Column does not match this parameter")
                dvec = d[:, basis_idx, 0, param_kind]

            else:                                 # pair → two columns per basis
                base = c_cur + 2 * basis_idx
                if not (base <= col_idx < base + 2):
                    raise ValueError("Column does not match this parameter")
                func_pos = col_idx - base         # 0 or 1
                dvec = d[:, basis_idx, func_pos, param_kind]

            return dvec if x_weights is None else x_weights * dvec

        # advance to next block
        p_cur += p_block
        c_cur += c_block

    raise ValueError("param_idx or col_idx out of range")
def parameter_index_to_function_mapping(function_types_and_amounts):
    mapping = []
    col_index = 0

    for ftype, amount in function_types_and_amounts:
        meta = _function_type_metadata[ftype]
        n_params = meta["num_params"]
        n_funcs = meta["num_functions"]

        for p in range(n_params):
            for i in range(amount):
                func_start = col_index + i * n_funcs
                mapping.append(list(range(func_start, func_start + n_funcs)))

        col_index += amount * n_funcs

    return mapping
def solve_linear_coefficients(X,y):
        """
        Solve the linear system X * coeffs = y for coeffs.
        """
        U, Sigma, Vh = xl.svd(X, full_matrices=False)
        y_goal= y
        sigma_inv = np.zeros_like(Sigma)
        nonzero_indices = Sigma > 1e-8  # Avoid division by zero
        sigma_inv[nonzero_indices] = 1 / Sigma[nonzero_indices]
        sigma_inv=np.diag(sigma_inv)
        psuedoinverse= Vh.T @ sigma_inv @ U.T
        c=psuedoinverse @ y_goal
        return c
class function(ABC):
    @abstractmethod
    def evaluate(self,x):
        pass

    @abstractmethod
    def evaluate_derivs(self, x):
        pass
class realGaussians_unshifted(function):
    def __init__(self, avals):
        self.function_shape=1 #One function per set of parameters
        self.param_count=1 #One parameter per function
        self.deriv_shape=1 #One derivative per function
        self.avals=np.asarray(avals,dtype=data_type)  # Ensure avals is a numpy array of doubles
        self.xsq = None
    def evaluate(self, x):
        if self.xsq is None:
            self.xsq = x**2
        xsq= self.xsq[:, None]
        return np.exp(-self.avals**2*xsq )[:,:,None]  # Shape: (len(x), len(self.avals), 1)
    def evaluate_derivs(self, x):
        if self.xsq is None:
            self.xsq = x ** 2
        xsq = self.xsq[:, None]
        retval=(-2 * self.avals * xsq * np.exp(-self.avals**2*xsq ))
        return retval[:,:,None,None]
class complexGaussians_unshifted(function):
    def __init__(self,avals,bvals):
        self.function_shape=1 #One function per set of parameters
        self.param_count=2
        self.deriv_shape=2
        self.avals=np.asarray(avals)
        self.bvals=np.asarray(bvals)
        self.xsq=None
    def evaluate(self, x):
        if self.xsq is None:
            self.xsq = x ** 2
        xsq= self.xsq[:, None]
        retval=np.exp(-(self.avals**2+1j*self.bvals)*xsq )
        return retval[:,:,None] #One function per set of parameters, shape: (len(x), len(self.avals), 1)
    def evaluate_derivs(self, x):
        if self.xsq is None:
            self.xsq = x ** 2
        xsq= self.xsq[:, None,None]
        func_eval= self.evaluate(x)
        #First, the derivative with respect to a
        avals= self.avals[None, :, None]
        da = -2 * avals * xsq * func_eval
        #Second, the derivative with respect to b
        db = -1j*xsq * func_eval
        return np.stack((da, db), axis=-1)  # Shape: (len(x), len(self.avals), 2, 1)
class complexGaussians_unshifted_pairs(function):
    def __init__(self, avals, bvals):
        self.function_shape = 2  # Two functions per set of parameters
        self.param_count = 2
        self.deriv_shape = 2
        self.avals = np.asarray(avals)
        self.bvals = np.asarray(bvals)
        self.xsq = None

    def evaluate(self, x):
        if self.xsq is None:
            self.xsq = x ** 2
        xsq= self.xsq[:, None]  # Reshape x to be a column vector
        exp_eval= np.exp(-(self.avals**2) * xsq)
        cos_term= np.cos(self.bvals * xsq)
        sin_term= np.sin(self.bvals * xsq)
        return np.stack((exp_eval * cos_term, exp_eval * sin_term), axis=-1)

    def evaluate_derivs(self, x):
        if self.xsq is None:
            self.xsq = x**2
        xsq = self.xsq[:, None]          # (N,1)

        exp_eval = np.exp(-(self.avals**2) * xsq)              # (N,m)
        cos_term = np.cos(self.bvals * xsq)                    # (N,m)
        sin_term = np.sin(self.bvals * xsq)                    # (N,m)

        # basis functions
        f_cos = exp_eval * cos_term                            # (N,m)
        f_sin = exp_eval * sin_term                            # (N,m)

        # ∂/∂a  (same sign for both columns)
        avals = self.avals[None, :]                            # (1,m)
        da = -2 * xsq * avals                                  # (N,m)
        da = np.stack((da * f_cos,                             # (N,m,2)
                       da * f_sin), axis=-1)

        # ∂/∂b  (sign differs between cos and sin columns)
        db_cos = -xsq * exp_eval * sin_term                    # (N,m)
        db_sin =  xsq * exp_eval * cos_term                    # (N,m)
        db = np.stack((db_cos, db_sin), axis=-1)               # (N,m,2)

        # final shape (N, m, 2 functions, 2 parameters)
        return np.stack((da, db), axis=-1)
class complexshiftedGaussians(function):
    """
    f_a,b,s,p(x) = exp( -(a² + i b)(x-s)² + i p (x-s) )

    parameters per basis element
    ----------------------------------
        a  : width          (real, >0)
        b  : imaginary part (real)
        s  : shift          (real)
        p  : momentum       (real)
    """
    def __init__(self, avals, bvals, shifts, momenta):
        self.function_shape = 1        # one column per (a,b,s,p)
        self.param_count    = 4
        self.deriv_shape    = 4

        self.avals   = np.asarray(avals, dtype=data_type)  # Ensure avals is a numpy array of doubles
        self.bvals   = np.asarray(bvals, dtype=data_type)  # Ensure bvals is a numpy array of doubles
        self.shifts  = np.asarray(shifts)
        self.momenta = np.asarray(momenta)

    # ------------------------------------------------------------------ #
    #  f(x)
    # ------------------------------------------------------------------ #
    def evaluate(self, x):
        x = np.asarray(x)
        t = x[:, None] - self.shifts[None, :]                      # (N,m)
        a2_ib = self.avals[None, :]**2 + 1j * self.bvals[None, :]  # (1,m)

        f = np.exp( -a2_ib * t**2 + 1j * self.momenta[None, :] * t )
        return f[:, :, None]                                       # (N,m,1)

    # ------------------------------------------------------------------ #
    #  ∂f/∂θ_k
    # ------------------------------------------------------------------ #
    def evaluate_derivs(self, x):
        x = np.asarray(x)
        t      = x[:, None] - self.shifts[None, :]                 # (N,m)
        a      = self.avals[None, :]                               # (1,m)
        b      = self.bvals[None, :]
        p      = self.momenta[None, :]
        a2_ib  = a**2 + 1j * b

        f = np.exp( -a2_ib * t**2 + 1j * p * t )                  # (N,m)

        # derivatives
        da = -2 * a * t**2 * f                                    # (N,m)
        db = -1j * t**2 * f                                       # (N,m)
        ds = ( 2*a2_ib * t - 1j * p ) * f                         # (N,m)
        dp = 1j * t * f                                           # (N,m)

        derivs = np.stack((da, db, ds, dp), axis=-1)              # (N,m,4)
        derivs = derivs[:, :, None, :]                            # (N,m,1,4)
        return derivs
class VarPro:
    def __init__(self,function_types_and_amounts,x,y,weights=None):
        self.function_types_and_amounts = function_types_and_amounts  # List of tuples (function_type, amount)
        self.x = np.asarray(x,dtype=data_type)
        self.y = np.asarray(y,dtype=data_type)
        self.weights = weights if weights is not None else np.ones_like(y)
        self.weights= np.asarray(self.weights, dtype=data_type)
        self.params = None  # Placeholder for the parameters to be optimized
        self.num_functions_total=0
        for func_type, amount in function_types_and_amounts:
            if func_type=="realGaussians_unshifted":
                self.num_functions_total+=amount
            elif func_type=="complexGaussians_unshifted":
                self.num_functions_total+=amount
            elif func_type=="complexGaussians_unshifted_pairs":
                self.num_functions_total+=amount*2
            else:
                raise ValueError(f"Unknown function type: {func_type}")
        pass
    def parse_params(self, params_dict):
        """
        Flatten `params_dict` into self.params (1-D numpy array). In particular, this means that first we take all a-values from the first function type, then all b-values from the first function type, then all a-values from the second function type, etc.
        """
        import numpy as np

        flat = []

        for func_type, amount in self.function_types_and_amounts:
            if func_type not in params_dict:
                raise KeyError(f"Missing entry for '{func_type}' in params_dict")
            for values in params_dict[func_type]:
                flat.extend(values)  # Flatten the values into the list
        self.params = np.asarray(flat, dtype=data_type)
        return self.params
    def unparse_params(self, p=None):

        # how many equally sized chunks we must carve out for each function type
        

        if p is None:
            if self.params is None:
                raise ValueError("No parameter vector supplied and self.params is None.")
            p = self.params

        p = np.asarray(p, dtype=data_type)
        params_dict = {}
        idx = 0  # running pointer in the flat vector

        for func_type, amount in self.function_types_and_amounts:
            k =  _function_type_metadata[func_type]["num_params"]
            if k is None:
                raise ValueError(f"Unknown function type: {func_type}")

            # carve out k consecutive chunks of length `amount`
            chunks = tuple(
                p[idx + i * amount : idx + (i + 1) * amount] for i in range(k)
            )
            idx += k * amount

            # store as single array or tuple of arrays, matching parse_params
            params_dict[func_type] = (chunks[0],) if k == 1 else chunks

        # sanity-check: did we consume exactly the whole vector?
        if idx != len(p):
            raise ValueError(
                f"Unparsed data left over: consumed {idx} elements, "
                f"but p has length {len(p)}."
            )

        return params_dict
    def setUpXmatrix(self, params_dict,x=None):
        if x is None:
            x=self.x
        N = len(x)
        M = self.num_functions_total
        X = np.empty((N, M), dtype=data_type)

        col = 0  # running column pointer

        for func_type, amount in self.function_types_and_amounts:

            # --- real Gaussians ------------------------------------------------
            if func_type == "realGaussians_unshifted":
                f = realGaussians_unshifted(params_dict[func_type])
                vals = f.evaluate(x)[:, :, 0]          # (N, amount)
                X[:, col : col + amount] = vals.astype(data_type)
                col += amount

            # --- complex Gaussians (single) -----------------------------------
            elif func_type == "complexGaussians_unshifted":
                f = complexGaussians_unshifted(*params_dict[func_type])
                vals = f.evaluate(x)[:, :, 0]          # (N, amount)
                X[:, col : col + amount] = vals
                col += amount

            # --- complex Gaussians (pairs → 2 per parameter set) --------------
            elif func_type == "complexGaussians_unshifted_pairs":
                f = complexGaussians_unshifted_pairs(*params_dict[func_type])
                vals = f.evaluate(x)                   # (N, amount, 2)
                vals = vals.reshape(N, amount * 2)          # flatten the “2”. This means that we first have 
                X[:, col : col + amount * 2] = vals.astype(data_type)
                col += amount * 2

            else:
                raise ValueError(f"Unknown function type: {func_type}")

        self.X = X
        return X
   
    def calculate_fitting_error(self,params):
        """
        Calculate the fitting error for the given parameters.
        """
        if isinstance(params, dict):
            # If params is a dictionary, parse it
            params_dict = params
            params = self.parse_params(params)
        elif isinstance(params, np.ndarray):
            # If params is a numpy array, unparse it
            params_dict = self.unparse_params(params)
        X = self.weights[:,None]*self.setUpXmatrix(params_dict)
        y_pred = X @ solve_linear_coefficients(X, self.weights*self.y)
        residuals = self.weights*self.y - y_pred
        error = (residuals.T @ residuals)

        return float(error)
    def calculate_fitting_error_and_derivative(self, params):
        """
        Calculate the fitting error and its derivative for the given parameters.
        """
        if isinstance(params, dict):
            # If params is a dictionary, parse it
            params_dict = params
            params = self.parse_params(params)
        elif isinstance(params, np.ndarray):
            # If params is a numpy array, unparse it
            params_dict = self.unparse_params(params)
        X = self.weights[:,None]*self.setUpXmatrix(params_dict)
        U, Sigma, Vh = xl.svd(X,full_matrices=False)
        y_goal= self.weights*self.y
        sigma_inv  = np.zeros_like(Sigma, dtype=data_type)

        nonzero_indices = Sigma >1e-8  # Avoid division by zero
        sigma_inv[nonzero_indices] = 1 / Sigma[nonzero_indices]
        sigma_inv=np.diag(sigma_inv)
        psuedoinverse= Vh.T @ sigma_inv @ U.T
        c=psuedoinverse @ y_goal
        y_pred= X @ c
        residuals = y_goal - y_pred
        error =(residuals.T @ residuals)

        derivative = np.zeros_like(params, dtype=data_type)



        #First, loop over the different function types
        #For vapro to work, I have to loop over the different parameters, and find out which functions they affect.
        function_mapping = parameter_index_to_function_mapping(self.function_types_and_amounts)
        derivs = [ _factory[ftype](params_dict[ftype]).evaluate_derivs(self.x)
           for ftype, _ in self.function_types_and_amounts ]
        for i,param in enumerate(params):
            #First, I have to figure out which function this parameter belongs to
             
            function_indices = function_mapping[i]
            Dk = np.zeros(self.X.shape, dtype=data_type)
            #Next, I need to find the derivative of the function with respect to this parameter
            for k in function_indices:
                dX_col = dX_column_for_param(i, k,self.function_types_and_amounts,derivs, x_weights=self.weights)
                Dk[:, k] = dX_col
            Dkc=Dk@c
            ak=Dkc - U@ (U.T @ Dkc)  # This is the derivative of the fitting error with respect to the parameter
            bk= U@ sigma_inv@(Vh@Dk.T@ residuals)  # This is the derivative of the fitting error with respect to the parameter
            #Now I have the jacobian
            jacobian_k=-(ak+ bk)
            derivative[i] = 2*np.real(jacobian_k.T @ residuals)  # This is the derivative of the fitting error with respect to the parameter
        return float(error), derivative.astype(np.float64, copy=False)

    def fit(self, initial_param_dict):

        # 1 · parse to *data_type* (internal high precision)
        initial_params_dd = self.parse_params(initial_param_dict)   # dtype=data_type

        # 2 · give SciPy a *float64* view
        x0 = initial_params_dd.astype(np.float64)

        # 3 · run the optimiser
        result = minimize(
            self.calculate_fitting_error_and_derivative,
            x0,
            jac=True,
            method="BFGS",
            options={"gtol": 1e-16},
        )

        # 4 · keep the final params in data_type for further use
        self.params = result.x.astype(data_type)

        self.linear_coefficients = solve_linear_coefficients(
            self.setUpXmatrix(self.unparse_params(self.params)), self.y
        )
        return self.unparse_params(self.params), result.fun    
    def eval_function(self,params,x):
        """
        Evaluate the function at the given x with the provided parameters.
        """
        if isinstance(params, dict):
            # If params is a dictionary, parse it
            params_dict = params
        elif isinstance(params, np.ndarray):
            # If params is a numpy array, unparse it
            params_dict = self.unparse_params(params)
        X = self.setUpXmatrix(params_dict,x)
        return X @ self.linear_coefficients
if __name__ == "__main__":
    x=np.logspace(-2,3,500)
    y=1/x
    weights=x
    num_real= 4
    num_complex= 1
    params_dict = {
        "realGaussians_unshifted": (np.logspace(-2,1, num_real),),
    }
    function_types_and_amounts = [(ftype,len(v[0])) for ftype, v in params_dict.items()]
    varpro = VarPro(function_types_and_amounts, x, y, weights)
    X_matrix= varpro.setUpXmatrix(params_dict)
    error= varpro.calculate_fitting_error(params_dict)
    best_params,error=varpro.fit(params_dict)
    error,deriv= varpro.calculate_fitting_error_and_derivative(best_params)
    print("Fitting error:", error)
    print("Derivative:", deriv)
    sys.exit(0)
