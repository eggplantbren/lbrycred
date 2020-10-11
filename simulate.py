import numpy as np
import numpy.random as rng

#rng.seed(0)
NUM_CHANNELS = 5
TOL = 1E-6

def Phi(x):
    """
    Softening function
    """
    return (x + 1)**0.25

def PhiInv(x):
    """
    Inverse
    """
    return x**4 - 1.0


def update(m, w):
    """
    One iteration of the procedure to compute the multipliers.
    Returns True if converged.
    """

    # Compute y from the current ms
    y = np.empty(NUM_CHANNELS)
    for j in range(NUM_CHANNELS):
        terms = m*w[:, j]
        terms[j] = w[j, j]
        y[j] = np.sum(terms)

    # Normalise y and compute updated m
    y *= np.sum(np.sum(w, axis=0))/np.sum(y)
    mnew = Phi(y)

    converged = np.mean(np.abs(mnew - m)) < TOL
    return mnew, converged

# Matrix of supports. Start with diagonal
w = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
w += np.diag(np.exp(3.0 + rng.randn(NUM_CHANNELS)))

# Add a few signed supports (if they land off-diagonal that's what they are)
for r in range(1 + rng.randint(10)):
    i = rng.randint(NUM_CHANNELS)
    j = rng.randint(NUM_CHANNELS)
    w[i, j] += np.exp(3.0 + rng.randn())

print("Support matrix:")
print("---------------")
print(w, end="\n\n")

print("Raw staked LBC:")
print("---------------")
print(np.sum(w, axis=0), end="\n\n")

# Credibilities
print("Calculating multipliers...", flush=True, end="")
m = np.ones(NUM_CHANNELS)
iterations = 0
while True:
    m, converged = update(m, w)
    iterations += 1
    if converged:
        break
print(f"converged after {iterations} iterations.", end="\n\n")

print("Final credibility scores on LBC grade:")
print("-----------------------------------")
y = PhiInv(m)
print(y)

