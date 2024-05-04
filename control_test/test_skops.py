import numpy as np

from skops.io import load
from skops.io import get_untrusted_types
        #with open(path + "beta1.pkl", "rb") as fp:   # Unpickling
        #    self.Beta1_grass = pickle.load(fp)

        #with open(path + "beta2.pkl", "rb") as fp:   # Unpickling
        #    self.Beta2_grass = pickle.load(fp)

unknown_types = get_untrusted_types(file="beta1.skops")
unknown_types = get_untrusted_types(file="beta2.skops")

Beta1_grass = load("beta1.skops", trusted=unknown_types)
Beta2_grass = load("beta2.skops", trusted=unknown_types)

V = 0.8
omega = 0.3

input = np.array([V,omega]).reshape((1,-1))

mu1, sigma1 = Beta1_grass.predict(input, return_std = True)
print(mu1)
print(sigma1)

mu2, sigma2 = Beta1_grass.predict(input, return_std = True)
print(mu2)
print(sigma2)