import numpy as np
import treams
import multiprocessing
import time
from threadpoolctl import threadpool_limits
freq = np.linspace(0.5, 1.5, 6*4)

def fun(freq):
    kpar = [0,0]
    lattice = treams.Lattice.square(0.5)
    pwbasis = treams.PlaneWaveBasisByComp.default(kpar)
    T = treams.TMatrix.sphere(6, 2*np.pi*freq, [0.2], [4, 1])
    Teff = T.latticeinteraction.solve(lattice, kpar)
    S = treams.SMatrices.from_array(Teff, pwbasis)
    illu = treams.plane_wave([0,0], 1, k0=2*np.pi*freq, basis=pwbasis, material=1, poltype='helicity')
    tr,ref = S.tr(illu)
    return tr

def test_multithreading():
    t1 = time.time()
    with threadpool_limits(limits=1, user_api='blas'):
        with multiprocessing.Pool(processes=6) as pool:
            xs = np.array(pool.map(fun, freq))
    t2 = time.time()
    reference = t2-t1

    t1 = time.time()
    with multiprocessing.Pool(processes=6) as pool:
        xs = np.array(pool.map(fun, freq))
    t2 = time.time()
    assert t2-t1 < 1.2*reference
