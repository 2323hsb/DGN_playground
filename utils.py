import numpy as np
from math import sin,cos,sqrt

# Generate ring distribution    
def generate_ring(sample_size=500):
    mean = [0, 0]
    cov = [[1,0],[0,1]]
    x, y = np.random.multivariate_normal(mean, cov, sample_size).T
    res_z = []
    for z in zip(x, y):
        z = np.array(z)
        res_z.append(z / 10 + z / np.linalg.norm(z))
    return np.array(res_z).astype('float32')

def generate_linear(sample_size=500):
    xx = np.array([-0.51, 51.2])
    yy = np.array([0.33, 51.6])
    means = [xx.mean(), yy.mean()]  
    stds = [xx.std() / 3, yy.std() / 3]
    corr = 0.8         # correlation
    covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
            [stds[0]*stds[1]*corr,           stds[1]**2]] 

    m = np.random.multivariate_normal(means, covs, sample_size)
    return m.astype('float32')

def gaussian_mixture(sample_size, n_dim=2, n_labels=2, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y, label]).reshape((3,))

    x = np.random.normal(0, x_var, (sample_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (sample_size, (int)(n_dim/2)))
    z = np.empty((sample_size, n_dim+1), dtype=np.float32)
    for batch in range(sample_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, 0:3] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)
    return z