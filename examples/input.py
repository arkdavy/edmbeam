#extinction distances
X0i = 1000  # nm
Xg = 58.0+ 1j * X0i * 1.06  # nm

# lattice parameter nm
a0 = 0.1

# Poisson's ratio
nu = 0.3

# crystal thickness, nm
t0 = 300  # nm

# the foil normal (also pointing into the image)
n = [5, 2, 8] #Miller indices
#nunit = n / (np.dot(n, n) ** 0.5)

# g-vector
g = [-1,1,1] #Miller indices

# deviation parameter (typically between -0.1 and 0.1)
s = 0.005

# electron beam direction (pointing into the image) DEFINES Z DIRECTION
z = [5, 1, 4] #Miller indices

# the dislocation Burgers vector (Miller Indices)
b0 = [-0.5, 0.0, -0.5]
#b0 = np.array((0.0, 0.5, -0.5))
#b0 = np.array((-0.5, -0.5, 0.0))
#b0 = np.array((eps, 0.0, 0.0))

# defect line direction
u = [5, 2, 3] #Miller indices

# integration step (decrease this until the image stops changing)
dt = 0.2  # fraction of a slice

# pixel scale is 1 nm per pixel, by default
pix2nm = 0.5 #nm per pixel

# pixels arounnd the dislocation 
pad = 20 #nm

#Gaussian blur sigma
blursigma = 2.0 #nm
