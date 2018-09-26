import numpy as np


class FactorAnalysisGM():
	def __init__(self, N, K, rho_b, rho_c, ):
		self.N = N
		self.K = K
		self.rho_b = rho_b 
		self.rho_c = rho_c
		


	def init_params():



	def sample(self, n_samples, return_comp):

		# sample from latent space
		z = np.random.normal(
			loc=0, 
			scale=1, 
			size=(self.K + 1, n_samples)
		)

		# transform to ambient space
		shared = np.dot(self.L, z)

		# draw private variability
		eps = np.random.multivariate_normal(
			mean=np.zeros(self.N + 1), 
			cov=self.Psi, 
			size=n_samples
		)
		# 
		X = shared + eps

		return X
