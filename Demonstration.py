import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from pba import Interval
from Useful import FuncPossibility, BalchStruct, nicegraph, CorrBounds, CorrBins
import json
# Naive propagation works by allowing errors from the estimation of one parameter to take up the slack in the estimation of another parameter.

# Example 1
## Sum of two probabilities guiding independent events

# distribution parameters
p1, p2 = 0.4, 0.7
n1, n2 = 10, 13

# Target Function
fun = lambda a, b: a+b
# Singh plots parameters
N = 1000

# Generate datasets
d1, d2 = sps.binom.rvs(p=p1, n=n1, size=N), sps.binom.rvs(p=p2, n=n2, size=N)

# Calculate confidence required to bound true value of p1+p2
alphas = [1-FuncPossibility([BalchStruct(d1[i], n1), BalchStruct(d2[i], n2)], fun(p1, p2), fun).left for i in range(N)]

## Plot the ECDF of the alpha levels
plt.plot(sorted(alphas), np.linspace(0,1,N), 'k', label = 'Alphas')

nicegraph(plt.gca())
plt.show()

# Example 2
## Sum of two probabilities guiding opposite dependent events
from Useful import CorrBins

# distribution parameters
p1, p2 = 0.4, 0.7
rho = -0.8
n = 15

# Target Function
fun = lambda a, b: a+b
# Singh plots parameters
N = 1000

# Generate datasets
data = [CorrBins(p1, p2, rho, n) for _ in range(N)]

# Calculate confidence required to bound true value of p1+p2
alphas = [1-FuncPossibility([BalchStruct(sum(data[i][0]), n), BalchStruct(sum(data[i][1]), n)], fun(p1, p2), fun).left for i in range(N)]

## Plot the ECDF of the alpha levels
plt.plot(sorted(alphas), np.linspace(0,1,N), 'k', label = 'Alphas')
nicegraph(plt.gca())
plt.show()

# Example 3
## Sum of two probabilities guiding positive dependent events

# distribution parameters
p1, p2 = 0.4, 0.7
rho = 0.5
n = 15

# Target Function
fun = lambda a, b: a+b
# Singh plots parameters
N = 1000

# Generate datasets
data = [CorrBins(p1, p2, rho, n) for _ in range(N)]

# Calculate confidence required to bound true value of p1+p2
alphas = [1-FuncPossibility([BalchStruct(sum(data[i][0]), n), BalchStruct(sum(data[i][1]), n)], fun(p1, p2), fun).left for i in range(N)]

## Plot the ECDF of the alpha levels
plt.plot(sorted(alphas), np.linspace(0,1,N), 'k', label = 'Alphas')

nicegraph(plt.gca())
plt.show()

# Example 4
## Sum of two probabilities, global Singh plot over different probabilities with extreme positive dependence


# Singh Plot parameters
N = 1000
GN = 200

# Target Function
fun = lambda a, b: a+b

# Generate probabilities
ps = np.random.rand(GN, 2)

# Fixed sample size
n = 15

# Generate extremal rhos
rhos = [CorrBounds(*pps).right-1e-9 for pps in ps]

# Generate Datasets
data = [[CorrBins(*ps[i], rhos[i], n) for _ in range(N)] for i in range(GN)]



# Read and write to json because calculating all these alphas is quite costly. But I encourage you to do it yourself for independent verification. If you just want to see a Singh plot then you can load from the saved data.

# Calculate confidence required to bound true value of p1+p2
### Uncomment this next line to generate it for yourself ###
# alphas = [sorted([1-FuncPossibility([BalchStruct(sum(data[j][i][0]), n), BalchStruct(sum(data[j][i][1]), n)], fun(*ps[j]), fun).left for i in range(N)]) for j in range(GN)]

# generated = {'data': data, 'alphas': alphas}
# np.save('GeneratedExample4.npy', generated)

### Comment out this line if you are generating the alpahs yourself ###
generated = np.load('GeneratedExample4.npy')

# Plot the Singh plot.
[plt.plot(A, np.linspace(0,1,N), alpha = 0.3) for A in alphas]
plt.plot(np.max(alphas, axis=0), np.linspace(0,1,N), 'k')

## Plot the cumulative distribution of the U(0,1) uniform for comparison
nicegraph(plt.gca())
plt.show()

# Example 5
## Various functions of two probabilities, global Singh plot over different probabilities with varying dependence

funs = [
    [lambda a, b: a+b, lambda a, b: a*b],
    [lambda a, b: a/b,
    lambda a, b: Interval(min(a.left, b.left), min(a.right, b.right)) if isinstance(a, Interval) else min(a,b)],
    [lambda a, b: a**b,
    lambda a, b: np.log(a)/np.log(b)]
]

funlabels = [
    [r'$a+b$', r'$ab$'],
    [r'$a/b$', r'$min(a,b)$'],
    [r'$a^b$', r'$\log_a b$']
]

# Singh Plot parameters
N = 1000
GN = 200

# Generate probabilities
ps = np.random.rand(GN, 2)

# Fixed sample size
n = 15

# Generate extremal rhos
rhos = [CorrBounds(*pps).width()*np.random.rand()+CorrBounds(*pps).left for pps in ps]

# Generate data
data = [[CorrBins(*ps[i], rhos[i], n) for _ in range(N)] for i in range(GN)]

# Read and write to json because calculating all these alphas is quite costly. But I encourage you to do it yourself for independent verification. If you just want to see a Singh plot then you can load from the saved data.

# Generate Datasets, uncomment this if you want to generate the values yourself.
alphas = []
for k, funss in enumerate(funs):
    alphas+=[[]]
    for l, funsss in enumerate(funss):
        # Calculate confidence required to bound true value of p1+p2
        alphas[-1] += [[sorted([1-FuncPossibility([BalchStruct(sum(data[j][i][0]), n), BalchStruct(sum(data[j][i][1]), n)], funsss(*ps[j]), funsss).left for i in range(N)]) for j in range(GN)]]

generated = {'data': data, 'alphas': alphas}
np.save('GeneratedExample5.npy', generated)

### Comment out this line if you are generating the alphas yourself ###
generated = np.load('GeneratedExample5.npy')

# Plot the Singh plots.
fig, ax = plt.subplots(3,2, figsize = [5, 8], sharey = 'row', sharex = 'col')
for k, funss in enumerate(funs):
    for l, funsss in enumerate(funss):
        [ax[k,l].plot(A, np.linspace(0,1,N), alpha = 0.3) for A in alphas[k][l]]
        ax[k,l].plot(np.max(alphas[k][l], axis=0), np.linspace(0,1,N), 'k')
        ax[k,l].set_xlim(0,1)
        ax[k,l].set_ylim(0,1)
        if k == 2: ax[k,l].set_xlabel(r'Confidence Level $1-\alpha$')
        if l == 0: ax[k,l].set_ylabel('Observed Rate of Coverage')
        ax[k,l].plot([0,1], [0,1], 'k:', label = r'U$(0,1)$')
        ax[k,l].set_title(funlabels[k][l])

plt.tight_layout()

plt.show()
