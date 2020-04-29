import unittest
import gpytorch
import numpy as np
import tempfile
import os
import warnings

from inference import gaussian_processes as gps

class TestCustomGP(unittest.TestCase):

    def assertAllClose(self,x,y,tolerance=2e-1):
        self.assertTrue(x.shape == y.shape)
        x = x.ravel()
        y = y.ravel()
        for k in range(len(x)):
            self.assertTrue(np.abs(x[k]-y[k])<tolerance)
    def test_initialization(self):
        train_x = np.linspace(0,1,9).reshape((3,3))
        train_y = np.sin(2*np.pi*train_x) + np.random.randn(9).reshape((3,3))*0.2

        mean_module = gpytorch.means.ConstantMean()
        prior = gpytorch.priors.NormalPrior(2,1)
        covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.CosineKernel(
                                period_length_prior = prior
                            )
                        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp = gps.CustomGP(train_x,train_y,likelihood,mean_module,covar_module)

        gp.covar_module.base_kernel.period_length = prior.mean

        self.assertEqual(gp.likelihood,likelihood)
        self.assertEqual(gp.mean_module,mean_module)
        self.assertEqual(gp.covar_module,covar_module)
        self.assertEqual(gp.covar_module.base_kernel.period_length.item(),2)

    def test_save_load(self):
        train_x = np.linspace(0,1,9).reshape((3,3))
        train_y = np.sin(2*np.pi*train_x) + np.random.randn(9).reshape((3,3))*0.2
        def get_blank_structure():
            mean_module = gpytorch.means.ConstantMean()
            covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.CosineKernel()
                            )
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            return likelihood,mean_module,covar_module

        gp = gps.CustomGP(train_x,train_y,*get_blank_structure())

        # We change a parameter to check that it is saved
        gp.covar_module.base_kernel.period_length = 100
        gp.likelihood.noise_covar.noise = 10

        save_file = tempfile.NamedTemporaryFile(suffix='.pth').name
        gp.save(save_file)

        self.assertTrue(os.path.isfile(save_file))

        loaded_gp = gps.CustomGP.load(save_file,train_x,train_y,*get_blank_structure())
        self.assertEqual(
                loaded_gp.covar_module.base_kernel.period_length,
                gp.covar_module.base_kernel.period_length
            )
        self.assertEqual(
            loaded_gp.likelihood.noise_covar.noise,
            gp.likelihood.noise_covar.noise
        )

    def test_optimization_1d(self):
        warnings.simplefilter('ignore',gpytorch.utils.warnings.GPInputWarning)

        x = np.linspace(0,1,101)
        y = x**2

        # prior = gpytorch.priors.NormalPrior(0.8, 1)
        mean_module = gpytorch.means.ConstantMean()
        covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.PolynomialKernel(power=2)
                        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        gp = gps.CustomGP(x,y,likelihood,mean_module,covar_module)
        # gp.mean_module.mean = prior.mean

        gp.optimize_hyperparameters(epochs=20)

        predictions = gp.infer(x)

        self.assertAllClose(y,predictions.mean.numpy())

    def test_optimization_2d(self):
        warnings.simplefilter('ignore',gpytorch.utils.warnings.GPInputWarning)

        x1 = np.linspace(0,1,11)
        x2 = np.linspace(0,1,11)
        x = np.dstack(np.meshgrid(x1,x2))
        x_l = x.reshape((-1,2))
        y = (x_l**2).sum(axis=1)#np.sin(x_l[:,0] * 2*np.pi - x_l[:,1] *2*np.pi)

        mean_module = gpytorch.means.ConstantMean()
        covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.PolynomialKernel(power=2)
                        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        gp = gps.CustomGP(x_l,y,likelihood,mean_module,covar_module)

        gp.optimize_hyperparameters(epochs=20)

        predictions = gp.infer(x_l)

        means = predictions.mean

        self.assertAllClose(y,predictions.mean.numpy())

class TestMaternKernelGP(unittest.TestCase):
    def test_initialization(self):
        train_input = np.linspace(0,1,101)
        train_target = np.sin(2*np.pi*train_input) + np.random.randn(len(train_input))*0.2

        gp = gps.MaternKernelGP(train_input,
                            train_target,
                            likelihood_prior=(1,0.5),
                            likelihood_noise_constraint=(1e-3,1e4),
                            lengthscale_prior=(2,1),
                            lengthscale_constraint=(2,4),
                            outputscale_constraint=(5,6.))

        self.assertTrue(isinstance(gp.likelihood,gpytorch.likelihoods.Likelihood))
        self.assertTrue(isinstance(gp.mean_module,gpytorch.means.Mean))
        self.assertTrue(isinstance(gp.covar_module,gpytorch.kernels.Kernel))

    def test_save_load(self):
        train_input = np.linspace(0,1,101)
        train_target = np.sin(2*np.pi*train_input) + np.random.randn(len(train_input))*0.2
        gp = gps.MaternKernelGP(train_input,
                            train_target,
                            likelihood_prior=(1,0.5),
                            likelihood_noise_constraint=(1e-3,1e4),
                            lengthscale_prior=(2,1),
                            lengthscale_constraint=(2,4),
                            outputscale_constraint=(5.,6.))
        save_file = tempfile.NamedTemporaryFile(suffix='.pth').name
        gp.save(save_file)

        loaded = gps.MaternKernelGP.load(save_file,train_input,train_target)

        self.assertEqual(gp.construction_parameters,loaded.construction_parameters)
        self.assertEqual(gp.likelihood.noise_covar.noise,loaded.likelihood.noise_covar.noise)
        self.assertEqual(gp.mean_module.constant,loaded.mean_module.constant)
        self.assertEqual(gp.covar_module.base_kernel.lengthscale,loaded.covar_module.base_kernel.lengthscale)
        self.assertEqual(gp.covar_module.outputscale,loaded.covar_module.outputscale)


if __name__ == '__main__':
    unittest.main()
