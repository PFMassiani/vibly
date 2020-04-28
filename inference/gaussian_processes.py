import gpytorch
import torch
from numpy import ndarray

def ensure_tensor(x):
    if isinstance(x,torch.Tensor):
        return x
    elif isinstance(x,ndarray):
        return torch.from_numpy(x.astype(float))
    else:
        return torch.Tensor(x)

class CustomGP(gpytorch.models.ExactGP):
    def __init__(self,train_x,train_y,likelihood,mean_module,covar_module):
        train_x = ensure_tensor(train_x)
        train_y = ensure_tensor(train_y)
        super(CustomGP,self).__init__(train_x,train_y,likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self,x):
        x = ensure_tensor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def save(self,save_path):
        """
            Saves the model with .pth extension. Note that you will need to
            specify the structure of the uninitialized model to load it.
        """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        torch.save(self.state_dict(),save_path)

    @staticmethod
    def load(load_path,train_x,train_y,mean_module,covar_module,
            likelihood = None):
        train_x = ensure_tensor(train_x)
        train_y = ensure_tensor(train_y)
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        state_dict = torch.load(load_path)
        model = CustomGP(train_x,train_y,likelihood,mean_module,covar_module)
        model.load_state_dict(state_dict)
        return model,likelihood

    def optimize_hyperparameters(self,x,y,epochs,optimizer_function=None,
                                mll_fun=None,**optimizer_kwargs):
        x = ensure_tensor(x)
        y = ensure_tensor(y)
        if optimizer_function is None:
            optimizer_method = torch.optim.Adam
        if mll_function is None:
            mll_function = gpytorch.mlls.ExactMarginalLogLikelihood

        self.train()
        self.likelihood.train()

        optimizer = optimizer_method((
                            [{'params':model.parameters()}],
                            **optimizer_kwargs
                        ))
        mll = mll_function(self.likelihood,self)

        for n in range(epochs):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output,y)
            loss.backward()
            optimizer.step()
