import importlib.metadata

import numpy as np

import ptsa

try:
    import h5py
except ImportError:
    h5py = None

try:
    import gmsh
except ImportError:
    gmsh = None

LENGTHS = {
    "ym": 1e-24,
    "zm": 1e-21,
    "am": 1e-18,
    "fm": 1e-15,
    "pm": 1e-12,
    "nm": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "mm": 1e-3,
    "cm": 1e-2,
    "dm": 1e-1,
    "m": 1,
    "dam": 1e1,
    "hm": 1e2,
    "km": 1e3,
    "Mm": 1e6,
    "Gm": 1e9,
    "Tm": 1e12,
    "Tm": 1e12,
    "Pm": 1e15,
    "Em": 1e18,
    "Zm": 1e21,
    "Ym": 1e24,
}

INVLENGTHS = {
    r"ym^{-1}": 1e24,
    r"zm^{-1}": 1e21,
    r"am^{-1}": 1e18,
    r"fm^{-1}": 1e15,
    r"pm^{-1}": 1e12,
    r"nm^{-1}": 1e9,
    r"um^{-1}": 1e6,
    r"µm^{-1}": 1e6,
    r"mm^{-1}": 1e3,
    r"cm^{-1}": 1e2,
    r"dm^{-1}": 1e1,
    r"m^{-1}": 1,
    r"dam^{-1}": 1e-1,
    r"hm^{-1}": 1e-2,
    r"km^{-1}": 1e-3,
    r"Mm^{-1}": 1e-6,
    r"Gm^{-1}": 1e-9,
    r"Tm^{-1}": 1e-12,
    r"Pm^{-1}": 1e-15,
    r"Em^{-1}": 1e-18,
    r"Zm^{-1}": 1e-21,
    r"Ym^{-1}": 1e-24,
}

FREQUENCIES = {
    "yHz": 1e-24,
    "zHz": 1e-21,
    "aHz": 1e-18,
    "fHz": 1e-15,
    "pHz": 1e-12,
    "nHz": 1e-9,
    "uHz": 1e-6,
    "µHz": 1e-6,
    "mHz": 1e-3,
    "cHz": 1e-2,
    "dHz": 1e-1,
    "s": 1,
    "daHz": 1e1,
    "hHz": 1e2,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
    "THz": 1e12,
    "PHz": 1e15,
    "EHz": 1e18,
    "ZHz": 1e21,
    "YHz": 1e24,
    r"ys^{-1}": 1e24,
    r"zs^{-1}": 1e21,
    r"as^{-1}": 1e18,
    r"fs^{-1}": 1e15,
    r"ps^{-1}": 1e12,
    r"ns^{-1}": 1e9,
    r"us^{-1}": 1e6,
    r"µs^{-1}": 1e6,
    r"ms^{-1}": 1e3,
    r"cs^{-1}": 1e2,
    r"ds^{-1}": 1e1,
    r"s^{-1}": 1,
    r"das^{-1}": 1e-1,
    r"hs^{-1}": 1e-2,
    r"ks^{-1}": 1e-3,
    r"Ms^{-1}": 1e-6,
    r"Gs^{-1}": 1e-9,
    r"Ts^{-1}": 1e-12,
    r"Ps^{-1}": 1e-15,
    r"Es^{-1}": 1e-18,
    r"Zs^{-1}": 1e-21,
    r"Ys^{-1}": 1e-24,
}


def generate_mesh_spheres(
    radii, positions, savename, modelname="model1", meshsize=-1, meshsize_boundary=-1
):
    if gmsh is None:
        Exception("optional dependency 'gmsh' not found, cannot create mesh")

    if meshsize == -1:
        meshsize = np.max(radii) * 0.2
    if meshsize_boundary == -1:
        meshsize_boundary = np.max(radii) * 0.2

    gmsh.initialize()
    gmsh.model.add(modelname)

    spheres = []
    for i, (radius, position) in enumerate(zip(radii, positions)):
        tag = i + 1
        gmsh.model.occ.addSphere(*position, radius, tag)
        spheres.append((3, tag))
        gmsh.model.addPhysicalGroup(3, [i + 1], tag)
        # Add surfaces for other mesh formats like stl, ...
        gmsh.model.addPhysicalGroup(2, [i + 1], tag)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshsize)
    gmsh.model.mesh.setSize(
        gmsh.model.getBoundary(spheres, False, False, True), meshsize_boundary
    )

    gmsh.model.mesh.generate(3)
    gmsh.write(savename)
    gmsh.finalize()


def _translate_polarizations(pols, helicity=True):
    if helicity:
        names = ["negative", "positive"]
    else:
        names = ["magnetic", "electric"]
    return [names[i] for i in pols]


def _translate_polarizations_inv(pols):
    helicity = {"plus": 1, "positive": 1, "minus": 0, "negative": 0}
    parity = {"te": 0, "magnetic": 0, "tm": 1, "magnetic": 1}
    if pols[0].decode() in helicity:
        dct = helicity
    elif pols[0].decode() in parity:
        dct = parity
    else:
        raise ValueError(f"unrecognized polarization {pols[0]}")
    return [dct[i.decode()] for i in pols], pols[0] in helicity


def save_hdf5(
    savename,
    tmats,
    name,
    description,
    id,
    unit_k0=r"nm^{-1}",
    unit_length="nm",
    embedding_name="Embedding",
    embedding_description="",
    frequency_axis=None,
    mode="w-",
):
    if h5py is None:
        Exception("optional dependency 'h5py' not found, cannot create hdf5file")
    tmats = np.array(tmats)
    shape = tmats.shape
    tmat_first = tmats.ravel()[0]
    with np.nditer(
        [tmats] + [None] * 4,
        ["refs_ok"],
        [["readonly"]] + [["writeonly", "allocate"]] * 4,
        [None, float, complex, complex, complex],
    ) as it:
        for (tmat, k0, epsilon, mu, kappa) in it:
            tmat = tmat.item()
            if (
                np.any(tmat.l != tmat_first.l)
                or np.any(tmat.m != tmat_first.m)
                or np.any(tmat.pol != tmat_first.pol)
                or np.any(tmat.pidx != tmat_first.pidx)
            ):
                raise ValueError("non-matching T-matrix modes")
            if tmat.helicity != tmat_first.helicity:
                raise ValueError("non-matching basis sets")
            if np.any(tmat.positions.shape != tmat.positions.shape):
                raise ValueError("non-matching positions")
            k0[...] = tmat.k0
            epsilon[...] = tmat.epsilon
            mu[...] = tmat.mu
            kappa[...] = tmat.kappa
        k0s, epsilons, mus, kappas = it.operands[1:]

    tms = np.stack([tmat.t for tmat in tmats.flatten()]).reshape(
        shape + tmat_first.t.shape
    )
    positions = np.stack([tmat.positions for tmat in tmats.flatten()]).reshape(
        shape + tmat_first.positions.shape
    )

    if np.all(k0s == k0s.ravel()[0]):
        k0s = k0s.ravel()[0]
    if np.all(epsilons == epsilons.ravel()[0]):
        epsilons = epsilons.ravel()[0]
    if np.all(mus == mus.ravel()[0]):
        mus = mus.ravel()[0]
    if np.all(kappas == kappas.ravel()[0]):
        kappas = kappas.ravel()[0]
    if np.all(positions - tmat_first.positions == 0):
        positions = tmat_first.positions

    if frequency_axis is not None:
        k0slice = (
            (0,) * frequency_axis
            + (slice(k0s.shape[frequency_axis]),)
            + (0,) * (k0s.ndim - frequency_axis - 1)
        )
        k0s = k0s[k0slice]

    _write_hdf5(
        savename,
        id,
        name,
        description,
        tms,
        k0s,
        epsilons,
        mus,
        kappas,
        tmat_first.l,
        tmat_first.m,
        tmat_first.pol,
        tmat_first.pidx,
        positions,
        unit_k0,
        unit_length,
        embedding_name,
        embedding_description,
        tmat_first.helicity,
        frequency_axis,
        mode,
    )


def _write_hdf5(
    savename,
    id,
    name,
    description,
    tms,
    k0s,
    epsilons,
    mus,
    kappas,
    ls,
    ms,
    pols,
    pidxs,
    positions,
    unit_k0=r"nm^{-1}",
    unit_length="nm",
    embedding_name="Embedding",
    embedding_description="",
    helicity=True,
    frequency_axis=None,
    mode="w-",
):
    with h5py.File(savename, mode) as f:
        f.create_dataset("tmatrix", data=tms)
        f["tmatrix"].attrs["id"] = id
        f["tmatrix"].attrs["name"] = name
        f["tmatrix"].attrs["description"] = description

        f.create_dataset("k0", data=k0s)
        f["k0"].attrs["unit"] = unit_k0

        f.create_dataset("modes/l", data=ls)
        f.create_dataset("modes/m", data=ms)
        f.create_dataset(
            "modes/polarization",
            data=_translate_polarizations(pols, helicity=helicity),
        )
        f.create_dataset("modes/position_index", data=pidxs)
        f["modes/position_index"].attrs[
            "description"
        ] = """
            For local T-matrices each mode is associated with an origin. This index maps
            the modes to an entry in positions.
        """
        f.create_dataset("modes/positions", data=positions)
        f["modes/positions"].attrs[
            "description"
        ] = """
            The postions of the origins for a local T-matrix.
        """
        f["modes/positions"].attrs["unit"] = unit_length

        f["modes/l"].make_scale("l")
        f["modes/m"].make_scale("m")
        f["modes/polarization"].make_scale("polarization")
        f["modes/position_index"].make_scale("position_index")

        ndims = len(f["tmatrix"].dims)
        f["tmatrix"].dims[ndims - 2].label = "Scattered modes"
        f["tmatrix"].dims[ndims - 2].attach_scale(f["modes/l"])
        f["tmatrix"].dims[ndims - 2].attach_scale(f["modes/m"])
        f["tmatrix"].dims[ndims - 2].attach_scale(f["modes/polarization"])
        f["tmatrix"].dims[ndims - 2].attach_scale(f["modes/position_index"])
        f["tmatrix"].dims[ndims - 1].label = "Incident modes"
        f["tmatrix"].dims[ndims - 1].attach_scale(f["modes/l"])
        f["tmatrix"].dims[ndims - 1].attach_scale(f["modes/m"])
        f["tmatrix"].dims[ndims - 1].attach_scale(f["modes/polarization"])
        f["tmatrix"].dims[ndims - 1].attach_scale(f["modes/position_index"])

        embedding_path = "materials/" + embedding_name.lower()
        f.create_group(embedding_path)
        f[embedding_path].attrs["name"] = embedding_name
        f[embedding_path].attrs["description"] = embedding_description
        f.create_dataset(embedding_path + "/relative_permittivity", data=epsilons)
        f.create_dataset(embedding_path + "/relative_permeability", data=mus)
        if np.any(kappas != 0):
            f.create_dataset(embedding_path + "/chirality", data=kappas)

        f["embedding"] = h5py.SoftLink("/" + embedding_path)

        if frequency_axis is not None:
            f["k0"].make_scale("k0")
            f["tmatrix"].dims[frequency_axis].label = "Wave number"
            f["tmatrix"].dims[frequency_axis].attach_scale(f["k0"])
            if np.ndim(epsilons) != 0:
                f[embedding_path + "/relative_permittivity"].dims[
                    frequency_axis
                ].label = "Wave number"
                f[embedding_path + "/relative_permittivity"].dims[
                    frequency_axis
                ].attach_scale(f["k0"])
            if np.ndim(mus) != 0:
                f[embedding_path + "/relative_permeability"].dims[
                    frequency_axis
                ].label = "Wave number"
                f[embedding_path + "/relative_permeability"].dims[
                    frequency_axis
                ].attach_scale(f["k0"])
            if np.ndim(kappas) != 0 and np.any(kappas != 0):
                f[embedding_path + "/chirality"].dims[
                    frequency_axis
                ].label = "Wave number"
                f[embedding_path + "/chirality"].dims[frequency_axis].attach_scale(
                    f["k0"]
                )
            if np.ndim(positions) > 2:
                f["modes/positions"].dims[frequency_axis].label = "Wave number"
                f["modes/positions"].dims[frequency_axis].attach_scale(f["k0"])


def _convert_to_k0(x, xtype, xunit, k0unit=r"nm^{-1}"):
    c = 299792458.0
    k0unit = INVLENGTHS[k0unit]
    if xtype in ("freq", "nu"):
        xunit = FREQUENCIES[xunit]
        return 2 * np.pi * x / c * (xunit / k0unit)
    elif xtype == "omega":
        xunit = FREQUENCIES[xunit]
        return x / c * (xunit / k0unit)
    elif xtype == "k0":
        xunit = INVLENGTHS[xunit]
        return x * (xunit / k0unit)
    elif xtype == "lambda0":
        xunit = LENGTHS[xunit]
        return 2 * np.pi / (c * xunit * k0unit)
    raise ValueError(f"unrecognized frequency/wavenumber/wavelength type: {xtype}")


def _scale_position(scale, data, offset=0):
    scale_axis = [i for i, x in enumerate(data.dims) if scale in x.values()]
    ndim = data.ndim
    if scale_axis:
        if len(scale_axis) == 1:
            return ndim - scale_axis[0] - 1 - offset
        raise Exception("scale added to multiple axes")
    return ndim - 1 - offset


def _load_parameter(param, group, tmatrix, frequency, append_dim=0, default=None):
    if param in group:
        dim = _scale_position(group[param], tmatrix, offset=2)
        if dim == 0:
            dim = _scale_position(frequency, group[param]) + append_dim
        res = group[param][...]
        return res.reshape(res.shape + (1,) * dim)
    return default


def load_hdf5(filename, lengthunit="nm"):
    with h5py.File(filename, "r") as f:
        for freq_type in ("freq", "nu", "omega", "k0", "lambda0"):
            if freq_type in f:
                ld_freq = f[freq_type][...]
                break
        if "modes/positions" in f:
            k0unit = f["modes/positions"].attrs.get("unit", lengthunit) + r"^{-1}"
        k0s = _convert_to_k0(ld_freq, freq_type, f[freq_type].attrs["unit"], k0unit)
        k0_dim = _scale_position(f[freq_type], f["tmatrix"], offset=2)
        k0s = k0s.reshape(k0s.shape + (1,) * k0_dim)

        found_epsilon_mu = False
        epsilon = _load_parameter(
            "relative_permittivity",
            f["embedding"],
            f["tmatrix"],
            f[freq_type],
            k0_dim,
            default=None,
        )
        mu = _load_parameter(
            "relative_permeability",
            f["embedding"],
            f["tmatrix"],
            f[freq_type],
            k0_dim,
            default=None,
        )
        if epsilon is None and mu is None:
            n = _load_parameter(
                "refractive_index",
                f["embedding"],
                f["tmatrix"],
                f[freq_type],
                k0_dim,
                default=1,
            )
            z = _load_parameter(
                "relative_impedance",
                f["embedding"],
                f["tmatrix"],
                f[freq_type],
                k0_dim,
                default=1,
            )
            epsilon = n / z
            mu = n * z
        epsilon = 1 if epsilon is None else epsilon
        mu = 1 if mu is None else mu

        kappa = _load_parameter(
            "chirality", f["embedding"], f["tmatrix"], f[freq_type], k0_dim, default=0
        )

        if "positions" in f["modes"]:
            dim = _scale_position(f["modes/positions"], f["tmatrix"], offset=2)
            if dim == 0:
                dim = _scale_position(f[freq_type], f["modes/positions"]) + k0_dim
            positions = f["modes/positions"][...]
            positions = positions.reshape(
                positions.shape[:-2] + (1,) * dim + positions.shape[-2:]
            )
        else:
            positions = np.array([[0, 0, 0]])

        # l_incident
        polarizations, helicity = _translate_polarizations_inv(
            f["modes/polarization"][...]
        )
        modes = (f["modes/l"][...], f["modes/m"][...], polarizations)
        if "position_index" in f["modes"]:
            modes = (f["modes/position_index"][...],) + modes

        shape = f["tmatrix"].shape[:-2]
        positions_shape = positions.shape[-2:]
        k0s = np.broadcast_to(k0s, shape)
        epsilon = np.broadcast_to(epsilon, shape)
        mu = np.broadcast_to(mu, shape)
        kappa = np.broadcast_to(kappa, shape)
        positions = np.broadcast_to(positions, shape + positions_shape)

        res = np.empty(shape, object)
        for i in np.ndindex(*shape):
            i_tmat = i + (slice(f["tmatrix"].shape[-2]), slice(f["tmatrix"].shape[-1]))
            i_positions = i + (slice(positions_shape[0]), slice(positions_shape[1]))
            res[i] = ptsa.TMatrix(
                f["tmatrix"][i_tmat],
                k0s[i],
                epsilon[i],
                mu[i],
                kappa[i],
                positions[i_positions],
                helicity,
                modes,
            )
        return res