---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
  formats: md:myst
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---

# 2D-Array of Spheres via intermediate TMatrixC

Here we explore how to convert a chain of scatterers described by individual T-matrices into a T-matrix of cylindrical waves (`TMatrixC`). We continue to place these `TMatrixC`s on yet another lattice perpendicular to the previous chain to obtain a 2D array.
Alternatively the biperiodic Lattice interaction can be solved for in a single step. Nonetheless we use this example to provide insight into the verstility and composability of T-matrices in `treams`.

Let's start by importing the neccessary libraries.
```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import treams
```

And parametrizing the problem:
```{code-cell}
k0s = 2 * np.pi * np.linspace(1 / 600, 1 / 350, 100)
material_slab = 3
thickness = 10
material_sphere = (4, 1, 0.05)
pitch = 500
lattice = treams.Lattice.square(pitch)
radius = 100
lmax = mmax = 3
```

We will have to:

0. For each frequency (vacuum wavenumber) of interest:
1. Compute the individual T-matrices of the scatterers (in this case spheres)
2. Compute the T-matrix incorporating multiple scattering interactions with the rest of the chain (given a fixed Bloch-phase expressed as the component of the wavevector parallel to the chain $k_z$).
3. Compute the cylindrical waves that emerge as a coherent superposition of the periodic spherical waves and use those to construct a T-matrix in the basis of cylindrical waves.
4. Stack these cylindrical T-matrices once more in the $x$-direction perpendicular to the chains.
5. Construct S-matrices from these stacked `TMatrixC`s 
6. Embed in a stratified environment leveraging the plane-wave description contained in the S-matrices
7. Evaluate transmission and reflection.

```{code-cell}
tr = np.zeros((len(k0s), 2))
for i, k0 in enumerate(k0s):                                                  #0.
    kpar = [0, 0.3 * k0]

    spheres = treams.TMatrix.sphere(                                          #1.
        lmax, k0, radius, [material_sphere, 1]
    ).latticeinteraction.solve(pitch, kpar[0])                                #2.
    cwb = treams.CylindricalWaveBasis.diffr_orders(kpar[0], mmax, pitch, 0.02)
    spheres_cw = treams.TMatrixC.from_array(spheres, cwb)                     #3.
    chain_cw = spheres_cw.latticeinteraction.solve(pitch, kpar[1])            #4.

    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 0.02)       
    plw = treams.plane_wave(kpar, [1, 0, 0], k0=k0, basis=pwb, material=1)
    slab = treams.SMatrices.slab(thickness, pwb, k0, [1, material_slab, 1])
    dist = treams.SMatrices.propagation([0, 0, radius], pwb, k0, 1)
    array = treams.SMatrices.from_array(chain_cw, pwb)                        #5.
    total = treams.SMatrices.stack([slab, dist, array])                       #6.
    tr[i, :] = total.tr(plw)                                                  #7.
```


Let's now visualize the resulting transmission and reflection (intensities):
```{code-cell}
fig, ax = plt.subplots()
ax.set_xlabel("Frequency (THz)")
ax.plot(299792.458 * k0s / (2 * np.pi), tr[:, 0])
ax.plot(299792.458 * k0s / (2 * np.pi), tr[:, 1])
ax.legend(["$T$", "$R$"])
fig.show()
```
