import gpytorch
import torch
from numpy import ndarray

def ensure_tensor(x):
    if isinstance(x,torch.Tensor):
        return x
    elif isinstance(x,ndarray):
        return torch.from_numpy(x.astype(float)).float()
    else:
        return torch.Tensor(x)

class CustomGP(gpytorch.models.ExactGP):
    def __init__(self,train_x,train_y,likelihood,mean_module,covar_module):
        self.train_x = ensure_tensor(train_x)
        self.train_y = ensure_tensor(train_y)
        super(CustomGP,self).__init__(self.train_x,self.train_y,likelihood)
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
    def load(load_path,train_x,train_y,likelihood,mean_module,covar_module):
        train_x = ensure_tensor(train_x)
        train_y = ensure_tensor(train_y)

        state_dict = torch.load(load_path)
        model = CustomGP(train_x,train_y,likelihood,mean_module,covar_module)
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
