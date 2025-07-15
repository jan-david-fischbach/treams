import treams 
import numpy as np
import matplotlib.pyplot as plt
import treams.special as sc

def test_farfields():
    positions = np.array([[0, 0, 0]])
    b = treams.SphericalWaveBasis.default(1, nmax=len(positions), positions=positions)
    b = b[b.pol==1]

    theta = np.linspace(0, np.pi, 61)
    phi = np.linspace(0, 2 * np.pi, 51)
    TP = np.meshgrid(theta, phi)
    theta_phis = np.array(TP).T.reshape(-1, 2)
    print(theta_phis)
    
    k0 = 1
    material = (1, 1, 0)
    sca = treams.spherical_wave(l=1, m=0, pol=1, k0=k0, poltype="parity", material=material)
    sca_sph = sca.expand(b, modetype="singular")

    sca_sph[b.m==0] = 0
    sca_sph[b.m==1] = 1
    sca_sph[b.m==-1] = 1

    ff = sca_sph.efarfield(theta_phis)
    ff_mesh = np.array(ff).reshape((*TP[0].shape, 3), order="F")

    print(f"Far field shape: {ff_mesh.shape}, {np.linalg.norm(ff_mesh, axis=-1).shape=}")

    Theta, Phi = TP             
    plt.pcolormesh(Phi/np.pi, Theta/np.pi, np.linalg.norm(ff_mesh, axis=-1))
    plt.xlabel(r"$\varphi$ [$\pi$]")
    plt.ylabel(r"$\theta$ [$\pi$]")
    plt.title("Far Field Spherical Wave")
    plt.colorbar(label="Magnitude")
    plt.savefig("tests/integration/far_field_spherical_wave.png")
    plt.close()

def test_vsw_M_ff():
    l = [2]
    m = [2]

    theta = np.linspace(0, np.pi, 61)
    phi = np.linspace(0, 2 * np.pi, 71)
    TP = np.meshgrid(theta, phi)
    theta_phis = np.array(TP).T.reshape(-1, 2)
    print(theta_phis)
    thetas = theta_phis[:, 0]
    phis = theta_phis[:, 1]
    print(f"{phis=}")
    res = sc.vsw_M_ff(l, m, thetas, phis)
    
    ff = res.reshape((*TP[0].shape, 3), order="F")
    Theta, Phi = TP             
    plt.pcolormesh(Phi/np.pi, Theta/np.pi, np.real(ff[:,:,2]), cmap="RdBu")
    plt.title("Single Component Far Field")
    plt.colorbar(label=r"$\Re\{E_\varphi\}$")
    plt.savefig("tests/integration/far_field_M.png")
    plt.close()
    
if __name__ == "__main__":
    test_farfields()
    test_vsw_M_ff()