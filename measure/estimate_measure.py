import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import GPy

import numpy as np
# from slippy.slip import *
import viability as vibly # TODO: get rid of this dependency
import inference
from scipy.stats import norm

class GenericMeasureEstimation:
    def __init__(self,state_grid,action_grid,gp,seed=None):
        np.random.seed(seed)

        self.gp = gp
        self.state_grid = state_grid
        self.action_grid = action_grid
        self.state_dim = len(self.state_grid)
        self.action_dim = len(self.action_grid)

        Q_grid = np.meshgrid(*(self.state_grid), *(self.action_grid), indexing='ij')
        self.Q_list = np.vstack(map(np.ravel, Q_grid)).T
        self.Q_shape = Q_grid[0].shape

    @property
    def input_dim(self):
        return self.action_dim + self.state_dim
    # The failure value is chosen such that at the point there is only some
    # probability left that the point is viable
    @property
    def failure_value(self):
        return - 2*np.sqrt(self.gp.likelihood.noise)

    def _sample_from_with_condition(self,max_n_points,to_sample,sampleable):
        idx_sampleable = np.argwhere(sampleable)
        n_points = min(max_n_points,len(idx_sampleable))
        idx_sample = np.random.choice(idx_sampleable,n_points,replace=False)
        return idx_sample

    def learn_hyperparameters(self,Q_M_knowledge,Q_V_knowledge,save_file,
                            epochs,optimizer_function,**optimizer_kwargs):
        Q_M_list = Q_M_knowledge.ravel().T
        is_viable = Q_V_knowledge.ravel().T

        # TODO Check how the threshold on the number of samples can be
        # removed/augmented with the parallelism made possible by GPyTorch
        idx = self._sample_from_with_condition(
                            max_n_points = 2000,
                            to_sample = Q_M_list,
                            sampleable = is_viable
                        )

        x_train = self.Q_list[idx,:]
        y_train = Q_M_list[idx,:]

        previous_x = self.gp.train_x
        previous_y = self.gp.train_y
        self.gp.set_data(x_train,y_train)

        self.gp.optimize_hyperparameters(
                                epochs = epochs,
                                optimizer_function = optimizer_function,
                                **optimizer_kwargs
                            )

        self.gp.save(save_file)
        self.gp.set_data(previous_x,previous_y)

    def project_Q2S(self, Q):
        action_axes = tuple(range(Q.ndim - self.action_dim, Q.ndim))
        return np.mean(Q, action_axes)

    def safe_level_set(self, safety_threshold = 0, confidence_threshold = 0.5,
                        current_state=None):
        if current_state is None:
            Q_est, Q_est_s2 = self.gp.infer(self.Q_list)
        else:
            a_grid = np.meshgrid(*(self.action_grid), indexing='ij')
            a_points = np.vstack(map(np.ravel, a_grid)).T

            # TODO:  check math
            state_points = np.ones(
                                (a_points.shape[0], len(self.action_grid]))
                            ) * current_state.T

            x_points = np.hstack((state_points, a_points))
            Q_est, Q_est_s2 = self.gp.infer(x_points)



        Q_level_set = norm.cdf((Q_est - safety_threshold) / np.sqrt(Q_est_s2))

        if confidence_threshold is not None:
            Q_level_set[np.where(Q_level_set < confidence_threshold)] = 0
            Q_level_set[np.where(Q_level_set > confidence_threshold)] = 1

        # TODO: Return boolean or int
        if current_state is None:
            return self.prediction_to_grid(Q_level_set)
        else:
            return Q_level_set.reshape(self.Q_shape[-self.action_dim:])
    # TODO unite Q_M and safe_level_set
    def Q_M(self, current_state = None):
        if current_state is None:
            Q_est, Q_est_s2 = self.gp.infer(self.Q_list)
        else:
            a_grid = np.meshgrid(*(self.action_grid), indexing='ij')
            a_points = np.vstack(map(np.ravel, a_grid)).T

            # TODO:  check math
            state_points = np.ones(
                                (a_points.shape[0], len(self.action_grid]))
                            ) * current_state.T

            x_points = np.hstack((state_points, a_points))
            Q_est, Q_est_s2 = self.gp.infer(x_points)

        if current_state is None:
            return self.prediction_to_grid(Q_est), self.prediction_to_grid(Q_est_s2)
        else:
            return Q_est.reshape(self.Q_shape[-self.action_dim:]), Q_est_s2.reshape(self.Q_shape[-self.action_dim:])

    def prediction_to_grid(self, pred):
        return pred.reshape(self.Q_shape)

class MeasureEstimation(GenericMeasureEstimation):

    def __init__(self, x_seed,y_seed, state_grid, action_grid, seed=None):
        # TODO change so it is a subclass of GenericMeasureEstimation
        # Don't forget to empty the dataset after initialization

        # GenericMeasureEstimation(state_grid, action_grid, gp, seed)
        # Steps :
        # 1. Create estimation method with x_seed,y_seed, and be careful
        #       with the prior values
        # 2. Initialize GenericMeasureEstimation object
        # 3. If specified, learn hyperparameters
        # 4. Empty data

        # This is the default GP for this class. If you want another
        # default, please implement another subclass of GenericMeasureEstimation.
        # The values of the parameters are chosen to match the previous
        # implementation.

        # TODO Compute `ranges` without computing the whole Q_grid
        Q_grid = np.meshgrid(*(self.state_grid), *(self.action_grid), indexing='ij')
        input_dim = len(self.state_grid) + len(self.action_grid)
        ranges = [Q_grid[i].max() - Q_grid[i].min() for i in range(input_dim)]
        # TODO add mean value parameterization in gaussian_processes
        gp = inference.MaternKernelGP(
                                train_x = x_seed,
                                train_y = y_seed,
                                likelihood_prior=(0.001,0.001),
                                likelihood_noise_constraint=(1e-7,1e-3),
                                lengthscale_prior=(1,1),
                                lengthscale_constraint=None,
                                outputscale_prior=(1,1),
                                outputscale_constraint=(1e-3,1e4)
                            )

if __name__ == "__main__":
    ################################################################################
    # Load and unpack data
    ################################################################################
    infile = open('../data/slip_map.pickle', 'rb')
    data = pickle.load(infile)
    infile.close()

    Q_map = data['Q_map']

    Q_F = data['Q_F']
    x0 = data['x0']
    poincare_map = data['P_map']
    p = data['p']
    grids = data['grids']

    ################################################################################
    # Compute measure from grid for warm-start
    ################################################################################

    Q_V, S_V = vibly.compute_QV(Q_map, grids)

    S_M = vibly.project_Q2S(Q_V, grids, np.mean)
    # S_M = vibly.project_Q2S(Q_V, grids, np.mean)

    #S_M = S_M / grids['actions'][0].size
    Q_M = vibly.map_S2Q(Q_map, S_M, Q_V)
    plt.plot(S_M)
    plt.show()
    plt.imshow(Q_M, origin='lower')
    plt.show()

    ################################################################################
    # Create estimation object
    ################################################################################

    AS_grid = np.meshgrid(grids['actions'][0], grids['states'][0])
    estimation = MeasureEstimation(state_dim=1, action_dim=1, seed=1)

    # Uncomment if you want to learn the hyperparameters of the GP. This might take a while
    # estimation.learn_hyperparameter(AS_grid, Q_M, Q_V, save='./model/prior.npy')

    X_seed = np.atleast_2d(np.array([38 / (180) * np.pi, .45]))

    initial_measure = .2
    y_seed = np.array([[initial_measure]])

    estimation.init_estimator(X_seed, y_seed, load='./model/prior.npy')


    X_grid_1, X_grid_2 = np.meshgrid(grids['states'], grids['actions'])
    X_grid = np.column_stack((X_grid_1.flatten(), X_grid_2.flatten()))

    estimation.set_grid_shape(X_grid, Q_M.shape)

    # Start from an empty data set
    estimation.set_data_empty()

    X_1, X_2 = np.meshgrid(grids['actions'], grids['states'])
    X = np.column_stack((X_1.flatten(), X_1.flatten()))

    Q_V_est = estimation.safe_level_set(safety_threshold=0, confidence_threshold=0.6)
    S_M_est = estimation.project_Q2S(Q_V_est)
    plt.plot(S_M_est)
    plt.show()

    Q_M_mean, Q_M_S2 = estimation.Q_M()
    plt.imshow(Q_M_mean, origin='lower')
    plt.show()

    # estimation.set_data(X=X_seed, Y=y_seed)
    # Q_V_est = estimation.safe_level_set(safety_threshold=0, confidence_threshold=0.6)
    # S_M_est = estimation.project_Q2S(Q_V_est)
    # plt.plot(S_M_est)
    # plt.show()
