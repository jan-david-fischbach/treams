"""Packaging of treams."""
import os

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from setuptools.modified import newer_group
except ImportError:
    from distutils.dep_util import newer_group

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


class _reusing_build_ext(_build_ext):
    """build_ext that reuses up-to-date extensions in editable installs."""

    def build_extension(self, ext):
        """Copy the extension from the source tree if it is newer than its sources.

        Editable installs compile in a fresh temporary directory, which would otherwise
        rebuild every extension after any change.
        """
        if getattr(self, "editable_mode", False) and not os.environ.get(
            "CYTHON_COVERAGE", False
        ):
            build_py = self.get_finalized_command("build_py")
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)
            if os.path.exists(inplace_file) and not newer_group(
                ext.sources + ext.depends, inplace_file
            ):
                self.mkpath(os.path.dirname(regular_file))
                self.copy_file(inplace_file, regular_file)
                return
        super().build_extension(ext)


if os.name == "nt":
    link_args = [
        "-static-libgcc",
        "-static-libstdc++",
        "-Wl,-Bstatic,--whole-archive",
        "-lwinpthread",
        "-Wl,--no-whole-archive",
    ]
    compile_args = ["-DMS_WIN64"]

    class build_ext(_reusing_build_ext):
        """build_ext for Windows."""

        def finalize_options(self):
            """Set compiler to gcc."""
            super().finalize_options()
            self.compiler = "mingw32"

        # https://cython.readthedocs.io/en/latest/src/tutorial/appendix.html
        def build_extensions(self):
            """Add Windows specific compiler and linker arguments."""
            if self.compiler.compiler_type == "mingw32":
                for e in self.extensions:
                    e.extra_compile_args = compile_args
                    e.extra_link_args = link_args
            super().build_extensions()

else:
    build_ext = _reusing_build_ext


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    """Add c and c++ code to source archive."""
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


keys = {"include_dirs": [np.get_include()]}
compiler_directives = {"language_level": "3"}
if os.environ.get("CYTHON_COVERAGE", False):
    keys["define_macros"] = [("CYTHON_TRACE_NOGIL", "1")]
    compiler_directives["linetrace"] = True

extension_names = [
    "treams.coeffs",
    "treams.config",
    "treams.cw",
    "treams.pw",
    "treams.sw",
    "treams.lattice._dsum",
    "treams.lattice._esum",
    "treams.lattice._gufuncs",
    "treams.lattice._misc",
    "treams.lattice.cython_lattice",
    "treams.special._bessel",
    "treams.special._coord",
    "treams.special._gufuncs",
    "treams.special._integrals",
    "treams.special._misc",
    "treams.special._ufuncs",
    "treams.special._waves",
    "treams.special._wigner3j",
    "treams.special._wignerd",
    "treams.special.cython_special",
]
extensions = [
    Extension(name, [f"src/{name.replace('.', '/')}.pyx"], **keys)
    for name in extension_names
]

if cythonize is not None:
    try:
        extensions = cythonize(extensions, compiler_directives=compiler_directives)
    except ValueError:
        extensions = no_cythonize(extensions)
else:
    extensions = no_cythonize(extensions)

setup(ext_modules=extensions, cmdclass={"build_ext": build_ext})
