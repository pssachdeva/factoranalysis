import numpy as np

from sklearn.decomposition import FactorAnalysis

class FactorAnalysisGM():
	def __init__(self, N, K, rho_b, rho_c, blocks=None):
		self.N = N
		self.K = K
		self.rho_b = rho_b 
		self.rho_c = rho_c
		self.blocks = blocks
		self.init_params()

	def init_params(self):
		# randomly choose variances
		self.variances = np.random.uniform(low=5, high=10, size=(self.N))

		# start by identifying the basis vectors for shared variability
		L = np.zeros((self.N, self.K + 1))

		# one basis vector will provide the background noise correlation
		L[:, -1] = np.sqrt(self.rho_b * self.variances)

		# split up the variances into the K clusters of correlated neurons
		corr_clusters = np.array_split(np.sqrt(self.variances), self.K)
		# we'll also split up an array of indices for bookkeeping
		indices = np.array_split(np.arange(self.N), self.K)

		# iterate over the K clusters, each of which will set a latent factor
		for lf_idx, group in enumerate(zip(corr_clusters, indices)):
			# split up the zipped iterator into the cluster and corresponding indices
			cluster_vars, cluster_idx = group
			# place the correct values in the latent factor
			L[cluster_idx, lf_idx] = np.sqrt(self.rho_c - self.rho_b) * cluster_vars

		self.L = L
		self.psi = self.variances - np.diag(np.dot(self.L, self.L.T))
		self.covar = np.dot(self.L, self.L.T) + np.diag(self.psi)

		return

	def get_corr_matrix(self):
		stdev_matrix = np.diag(1./np.sqrt(np.diag(self.covar)))
		corr = np.dot(stdev_matrix, np.dot(self.covar, stdev_matrix))
		return corr

	def sample(self, n_samples):
		# sample from latent space
		z = np.random.normal(
			loc=0, 
			scale=1, 
			size=(self.K + 1, n_samples)
		)

		# transform to ambient space
		shared = np.dot(self.L, z).T

		# draw private variability
		eps = np.random.multivariate_normal(
			mean=np.zeros(self.N), 
			cov=np.diag(self.psi), 
			size=n_samples
		)
		# 
		X = shared + eps

		return X

	@staticmethod
	def ll(train, test, L, psi):
		'''
		L : n components (K) x n features (N)
		Psi : n features 

		'''
		# centering constant
		mean = np.mean(train, axis=0)

		# center the test data
		test = test - mean

		# calculate precision
		cov = np.dot(L.T, L)
		cov.flat[::len(cov)+1] += psi
		precision = np.linalg.inv(cov)

		# calculate ll
		first_term = -0.5 * np.sum(test * np.dot(test, precision), axis=1)
		second_term = -0.5 * np.log(np.linalg.det(2 * np.pi * cov))
		ll = np.mean(first_term + second_term)
	
		return ll