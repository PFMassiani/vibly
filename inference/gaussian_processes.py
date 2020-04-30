import gpytorch
import torch
from numpy import ndarray
import itertools
from numbers import Number

def class_from_name(name):
    return globals().get(name)

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
        return gpytorch.constraints.GreaterThan(max(0.,float(bounds)))
    else:
        lbound = max(0.,float(bounds[0]))
        ubound = max(0.,float(bounds[1]))
        if ubound - lbound < 1e-4:
            warnings.warn(('The Interval constraint has bounds that are very close.'
                            'Make sure that this is the intended behaviour.\n'
                            'Interval : [{:.3f} ; {:.3f}]'.format(lbound,ubound)))
        return gpytorch.constraints.Interval(lbound,ubound)

class CustomGP(gpytorch.models.ExactGP):
    CONSTRUCTION_INFORMATION_SAVE_KEY = "__construction_information__"
    TYPENAME_SAVE_KEY = "__typename__"

    def __init__(self,train_x,train_y,likelihood=None,mean_module=None,
                covar_module=None,likelihood_noise_constraint=(1e-3,1e4)):
        """
            Parameters:
                train_x : np.ndarray/torch.Tensor
                train_y : np.ndarray/torch.Tensor
                likelihood : gpytorch.likelihoods.Likelihood / tuple
                    If a gpytorch.likelihoods.Likelihood, is used to initialize
                    the GP's likelihood
                    If None, the likelihood is initialized as a default
                    GaussianLikelihood.
                    If a tuple, the likelihood is initialized as a GaussianLikelihood
                    with prior NormalPrior(*likelihood)
                mean_module : gpytorch.means.Mean / None
                    If None, the mean_module is initialized as a ConstantMean.
                covar_module : gpytorch.kernel.Kernel / None
                    If None, the covar_module is initialized as a
                    ScaleKernel(RBFKernel()).
                likelihood_noise_constraint : tuple / float / None
                    The bounds in which to constrain the noise level of the
                    likelihood
        """
        self.train_x = ensure_tensor(train_x)
        self.train_y = ensure_tensor(train_y)
        if isinstance(likelihood,gpytorch.likelihoods.Likelihood):
            pass
        else:
            prior = gpytorch.priors.NormalPrior(*likelihood) if likelihood is not None else None
            constraint = get_bound_constraint(likelihood_noise_constraint)

            likelihood = gpytorch.likelihoods.GaussianLikelihood(prior,constraint)

            if prior is not None:
                likelihood.noise_covar.noise = prior.mean


        super(CustomGP,self).__init__(self.train_x,self.train_y,likelihood)
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

    def save(self,save_path,additional_save_information=None):
        """
            Saves the model with .pth extension. Note that you will need to
            specify the structure of the uninitialized model to load it.
            The best practice is to store the construction parameters in
            additional_save_information.
            Note that you will NOT be able to load a saved model in the
            following cases:
                * The method for constructing it has changed since last saving
                * Some structure has been added to the model after construction
                    To add structure to a model, you should declare a subclass
                    of CustomGP.
        """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        save_dict = {'state_dict':self.state_dict()}
        # The user may want to save additional information with the weights.
        # It is actually very useful in order to be able to reconstruct the
        # object when loading it : PyTorch does not save the structure of the
        # model (due to the shortcoming of pickles when serializing classes),
        # so we need to have the name of the constructor and the parameters it
        # needs to reconstruct it.
        if additional_save_information is not None:
            save_dict[CustomGP.CONSTRUCTION_INFORMATION_SAVE_KEY] = additional_save_information
        save_dict[CustomGP.TYPENAME_SAVE_KEY] = type(self).__name__

        torch.save(save_dict,save_path)
        return save_path

    @staticmethod
    def load(load_path,train_x,train_y,likelihood=None,mean_module=None,
            covar_module=None):
        """
            If all of the optional parameters are specified, this method will
            attempt to load a CustomGP from load_path. Note that the saved model
            needs to have the exact same structure as the one you specify.
            If any of the optional parameters is None, this method will infer
            the type of CustomGP saved at load_path, and will attempt to
            initialize it with the construction information saved in the file.
            Remarks:
                * If the signature of the constructor of the saved GP has changed
                    since it has been saved, then this method will fail.
                * If the code that defines the saved GP is not in the file
                    gaussian_processes.py, then this method will fail.
                * If structure has been added to the saved model after its
                    construction (e.g., a prior has been added), then this
                    method will fail.
                    The correct way to add structure to a model is to create a
                    subclass of CustomGP
            Parameters:
                load_path : str : the path of the saved model
                train_x : np.ndarray / torch.Tensor
                train_y : np.ndarray / torch.Tensor
                likelihood : gpytorch.likelihoods.Likelihood : unused if the
                                                    saved model is not a CustomGP
                mean_module : gpytorch.means.Mean : unused if the saved model
                                                    is not a CustomGP
                covar_module : gpytorch.kernels.Kernel : unused if the saved
                                                    model is not a CustomGP
        """
        save_dict = torch.load(load_path)
        classname = save_dict[CustomGP.TYPENAME_SAVE_KEY]
        if classname != 'CustomGP' or (likelihood is None or
                                        mean_module is None or
                                        covar_module is None):
            constructor = class_from_name(classname)
            construction_parameters = save_dict[CustomGP.CONSTRUCTION_INFORMATION_SAVE_KEY]
            model = constructor(train_x=train_x,train_y=train_y,**construction_parameters)

            state_dict = save_dict['state_dict']
        else:
            train_x = ensure_tensor(train_x)
            train_y = ensure_tensor(train_y)

            model = CustomGP(train_x,train_y,likelihood,mean_module,covar_module)

            state_dict = torch.load(load_path)['state_dict']

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
            output = self(self.train_x)
            loss = -mll(output,self.train_y)
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
    def __init__(self,train_x,train_y,likelihood_prior=None,
                likelihood_noise_constraint=(1e-3,1e4),
                lengthscale_prior=(1,1),lengthscale_constraint=None,
                outputscale_constraint=(1e-3,1e4)):
        """
            Parameters:
                train_x : np.ndarray/torch.Tensor
                train_y : np.ndarray/torch.Tensor
                likelihood_prior : tuple / None
                    If None, the likelihood is initialized as a default
                    GaussianLikelihood.
                    If a tuple, the likelihood is initialized as a
                    GaussianLikelihood with prior
                    NormalPrior(likelihood[0],likelihood[1]).
                    Note that if you pass a GPyTorch Likelihood object, the
                    construction will not fail but you may not be able to load
                    the model after saving it.
                likelihood_noise_constraint : tuple / float / None
                    See CustomGP.__init__'s documentation
                lengthscale_prior : tuple / None
                    If None, the lengthscale has no prior
                    If tuple of size 2, the lengthscale has NormalPrior
                    initialized with the tuple
                lengthscale_constraint: float/tuple / None
                    If a tuple, the lengthscale of the Matern kernel is
                    constrained to stay between the given values.
                    If a float, the lengthscale is constrained to be greater
                    than max(0,lengthscale_constraint)
                    If None, the lengthscale is contrained to be positive.
                outputscale_constraint: float/tuple/None
                    Same, for the outputscale
        """
        self.construction_parameters = {
            'likelihood_prior' : likelihood_prior,
            'likelihood_noise_constraint' : likelihood_noise_constraint,
            'lengthscale_prior' : lengthscale_prior,
            'lengthscale_constraint': lengthscale_constraint,
            'outputscale_constraint': outputscale_constraint
        }

        mean_module = gpytorch.means.ConstantMean()

        if lengthscale_prior is not None:
            lengthscale_prior = gpytorch.priors.NormalPrior(*lengthscale_prior)
        lengthscale_constraint = get_bound_constraint(lengthscale_constraint)
        outputscale_constraint = get_bound_constraint(outputscale_constraint)
        ard_num_dims = None if len(train_x.shape) == 1 else train_x.shape[1]
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu = 5/2,
                ard_num_dims = ard_num_dims,
                lengthscale_prior = lengthscale_prior,
                lengthscale_constraint = lengthscale_constraint
            ),
            outputscale_constraint = outputscale_constraint
        )
        if lengthscale_prior is not None:
            covar_module.base_kernel.lengthscale = lengthscale_prior.mean

        super(MaternKernelGP,self).__init__(train_x,train_y,likelihood_prior,
                                            mean_module,covar_module,
                                            likelihood_noise_constraint)

    def save(self,save_path):
        return super(MaternKernelGP,self).save(save_path,self.construction_parameters)
