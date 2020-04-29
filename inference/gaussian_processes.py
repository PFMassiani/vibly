import gpytorch
import torch
from numpy import ndarray
from numbers import Number
import itertools

def ensure_tensor(x):
    if isinstance(x,torch.Tensor):
        return x
    elif isinstance(x,ndarray):
        return torch.from_numpy(x.astype(float)).float()
    else:
        return torch.Tensor(x)
# bounds = (b1,b2)
# bounds = v
# bounds = None
def get_bound_constraint(bounds):
    if bounds is None:
        return gpytorch.constraints.Positive()
    elif isinstance(bounds,Number):
        return gpytorch.constraints.GreaterThan(max(0,float(bounds)))
    else:
        lbound = max(0,bounds[0])
        ubound = max(0,bounds[0])
        return gpytorch.constraints.Interval(lbound,ubound)

class CustomGP(gpytorch.models.ExactGP):
    def __init__(self,train_input,train_target,likelihood=None,mean_module=None,
                covar_module=None,likelihood_noise_bounds=(1e-3,1e4)):
        """
            Parameters:
                train_input : np.ndarray/torch.Tensor
                train_target : np.ndarray/torch.Tensor
                likelihood : gpytorch.likelihoods.Likelihood / tuple
                    If a gpytorch.likelihoods.Likelihood, is used to initialize
                    the GP's likelihood
                    If None, the likelihood is initialized as a default
                    GaussianLikelihood.
                    If a tuple, the likelihood is initialized as a GaussianLikelihood
                    with prior NormalPrior(likelihood[0],likelihood[1])
                mean_module : gpytorch.means.Mean / None
                    If None, the mean_module is initialized as a ConstantMean.
                covar_module : gpytorch.kernel.Kernel / None
                    If None, the covar_module is initialized as a
                    ScaleKernel(RBFKernel()).
                likelihood_noise_bounds : tuple / float / None
                    The bounds in which to constrain the noise level of the
                    likelihood
        """
        train_input = ensure_tensor(train_input)
        train_target = ensure_tensor(train_target)
        if isinstance(likelihood,gpytorch.likelihoods.Likelihood):
            pass
        else:
            prior = gpytorch.priors.NormalPrior(*likelihood) if likelihood is not None else None
            constraint = get_bound_constraint(likelihood_noise_bounds)

            likelihood = gpytorch.likelihoods.GaussianLikelihood(prior,constraint)

            
        super(CustomGP,self).__init__(train_input,train_target,likelihood)
        if mean_module is not None:
            self.mean_module = mean_module
        else:
            self.mean_module = gpytorch.means.ConstantMean()
        if covar_module is not None:
            self.covar_module = covar_module
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                                    gpytorch.kernels.RBFKernel()
                                )


    def forward(self,x):
        x = ensure_tensor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def save(self,save_path,additional_save_information=None,
            additional_save_key='additional_dict'):
        """
            Saves the model with .pth extension. Note that you will need to
            specify the structure of the uninitialized model to load it.
        """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        save_dict = self.state_dict().copy()
        if additional_save_information is not None:
            save_dict[additional_save_key] = additional_save_information
        torch.save(save_dict,save_path)

    @staticmethod
    def load(load_path,train_input,train_target,likelihood,mean_module,covar_module,
                model = None):
        if model is None:
            train_input = ensure_tensor(train_input)
            train_target = ensure_tensor(train_target)

            model = CustomGP(train_input,train_target,likelihood,mean_module,covar_module)

        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)
        return model

    def optimize_hyperparameters(self,epochs,optimizer_function=None,
                                mll_function=None,**optimizer_kwargs):
        if optimizer_function is None:
            optimizer_method = torch.optim.Adam
        if mll_function is None:
            mll_function = gpytorch.mlls.ExactMarginalLogLikelihood

        self.train()
        self.likelihood.train()

        optimizer = optimizer_method(
                            [{'params':self.parameters()}],
                            **optimizer_kwargs
                        )
        mll = mll_function(self.likelihood,self)

        for n in range(epochs):
            optimizer.zero_grad()
            output = self(self.train_input)
            loss = -mll(output,self.train_target)
            loss.backward()
            optimizer.step()

    def infer(self,x,gp_only=False):
        x = ensure_tensor(x)

        self.eval()
        self.likelihood.eval()

        with torch.no_grad(),gpytorch.settings.fast_pred_var():
            if gp_only:
                return self(x)
            else:
                return self.likelihood(self(x))

    def __call__(self,*args,**kwargs):
        args = map(ensure_tensor,args)
        kwargs = {k:ensure_tensor(v) for k,v in kwargs.items()}
        return super(CustomGP,self).__call__(*args,**kwargs)


class MaternKernelGP(CustomGP):
    def __init__(self,train_input,train_target,likelihood_prior=None,
                likelihood_noise_bounds=(1e-3,1e4),
                lengthscale_prior=(1,1),lengthscale_constraints=None,
                outputscale_constraints=(1e-3,1e4)):
        """
            Parameters:
                train_input : np.ndarray/torch.Tensor
                train_target : np.ndarray/torch.Tensor
                likelihood : tuple / None
                    If None, the likelihood is initialized as a default
                    GaussianLikelihood.
                    If a tuple, the likelihood is initialized as a GaussianLikelihood
                    with prior NormalPrior(likelihood[0],likelihood[1]).
                    Note that if you pass a GPyTorch Likelihood object, the
                    construction will not fail but you may not be able to load
                    the model after saving it.
                likelihood_noise_bounds : tuple / float / None
                    See CustomGP.__init__'s documentation
                lengthscale_prior : tuple / None
                    If None, the lengthscale has no prior
                    If tuple of size 2, the lengthscale has NormalPrior
                    initialized with the tuple
                lengthscale_constraints: float/tuple / None
                    If a tuple, the lengthscale of the Matern kernel is
                    constrained to stay between the given values.
                    If a float, the lengthscale is constrained to be greater
                    than max(0,lengthscale_constraints)
                    If None, the lengthscale is contrained to be positive.
                outputscale_constraints: float/tuple/None
                    Same, for the outputscale
        """
        mean_module = gpytorch.means.ConstantMean()

        if lengthscale_prior is not None:
            lengthscale_prior = gpytorch.priors.NormalPrior(*lengthscale_prior)
        lengthscale_constraints = get_bound_constraint(lengthscale_constraints)
        outputscale_constraints = get_bound_constraint(outputscale_constraints)

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu = 5/2,
                ard_num_dims = train_input.shape[1],
                lengthscale_prior = lengthscale_prior,
                lengthscale_constraint = lengthscale_constraints
            ),
            outputscale_constraint = outputscale_constraints
        )

        super(MaternKernelGP,self).__init__(train_input,train_target,likelihood,
                                            mean_module,covar_module,likelihood_noise_bounds)

        self.construction_parameters = {
            'likelihood' : likelihood,
            'likelihood_noise_bounds' : likelihood_noise_bounds,
            'lengthscale_prior' : lengthscale_prior,
            'lengthscale_constraints': lengthscale_constraints,
            'outputscale_constraints': outputscale_constraints
        }

    def save(self,save_path):
        super(MaternKernelGP,self).save(save_path,self.construction_parameters,
                                        'construction_parameters')

    @staticmethod
    def load(load_path,train_input,train_target):
        save_dict = torch.load(load_path)
        state_dict = save_dict['state_dict']
        construction_parameters = save_dict['construction_parameters']
        model = MaternKernelGP(train_input, train_target,**construction_parameters)
        model.load_state_dict(state_dict)
        return model
