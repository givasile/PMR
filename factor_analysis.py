from sklearn.datasets import load_digits
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import numpy as np


digits = load_digits()
X, _ = load_digits(return_X_y=True)

transformer = FactorAnalysis(n_components=7, random_state=0)
X_transformed = transformer.fit_transform(X)
# X_transformed.shape

# np.linalg.inv(transformer.components_X)

# plt.matshow(X[0].reshape((8, 8)))
# plt.show(block=False)


# plt.matshow(digits.images[0])
# plt.show(block=False)
