import numpy as np
import numpy.random as rng

NUM_CHANNELS = 4

# Softening function
def phi(x):
    return np.log10(phi + 10.0)

def update(y, w):
    """
    One iteration of the procedure to compute the ys
    """
    ynew = 0.0

# Matrix of supports. Start with diagonal
w = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
w += np.diag(np.exp(3.0 + rng.randn(NUM_CHANNELS)))

# Add a few signed supports (if they land off-diagonal that's what they are)
for r in range(3):
    i = rng.randint(NUM_CHANNELS)
    j = rng.randint(NUM_CHANNELS)
    w[i, j] += np.exp(3.0*rng.randn())

print("Support matrix:")
print("---------------", end="\n\n")
print(w, end="\n\n")

# Credibilities
y = np.ones(NUM_CHANNELS)
print("After 0 iterations:", y)
for i in range(10):
    update(y, w)
    print(f"After {i+1} iterations:", y)

