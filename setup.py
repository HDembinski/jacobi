from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import re

with open("src/jacobi/__init__.py") as f:
    version = re.search('__version__ = "([^"]+)"', f.read()).group(1)


ext_modules = [
    Extension(
        "_cpp",
        # Sort input source files to ensure bit-for-bit reproducible builds
        # (https://github.com/pybind/python_example/pull/53)
        sorted(["src/main.cpp"]),
        include_dirs=[
            # Path to pybind11 headers
            "extern/pybind11/include",
            "extern/boost-math/include",
        ],
        language="c++",
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support " "is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.define_macros = [
                ("VERSION_INFO", '"{}"'.format(self.distribution.get_version()))
            ]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name="jacobi",
    version=version,
    author="Hans Dembinski",
    author_email="hans.dembinski@gmail.com",
    url="https://github.com/hdembinski/jacobi",
    description="Numerical derivatives",
    long_description="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.5.0"],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
