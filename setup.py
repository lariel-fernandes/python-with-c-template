import json
import os
import subprocess
import sysconfig
from typing import Type, Callable

import numpy
import torch.utils.cpp_extension
from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension as BaseBuildExtension


def get_extension(
    path: str,
    source_file_types: set[str],
    ext_type: Type[Extension] | Callable = Extension,
    **kwargs,
) -> Extension:
    """Create extension from sources"""
    libs, pkg_configs = load_libs_and_pkg_configs(path)
    return ext_type(
        libraries=libs,
        sources=find_sources(path, source_file_types),
        include_dirs=list(set(get_include_dirs(pkg_configs) + kwargs.pop("include_dirs", []))),
        library_dirs=list(set(get_library_dirs(pkg_configs) + kwargs.pop("library_dirs", []))),
        **kwargs,
    )


def find_sources(path: str, source_file_types: set[str]) -> list[str]:
    """Find all source files recursively"""
    return [
        str(os.path.join(root, file))
        for root, dirs, files in os.walk(path)
        for file in files
        if any(file.endswith(x) for x in source_file_types)
    ]


def load_libs_and_pkg_configs(ext_path: str) -> tuple[list[str], list[str]]:
    """Gets the libs and pkg-config arguments for the specified extension"""
    packages = load_packages_json(ext_path)
    libs = list(packages.keys())
    pkg_configs = [pkg_config for details in packages.values() for pkg_config in details.get("pkg_config") or []]
    return libs, pkg_configs


def load_packages_json(ext_path: str) -> dict[str, dict[str, list[str]]]:
    """Load the packages.json file for the specified extension"""
    if os.path.exists(packages_file := os.path.join(ext_path, "packages.json")):
        with open(packages_file, "r") as f:
            return json.load(f)
    return {}


def get_include_dirs(packages: list[str]) -> list[str]:
    """Get include directories from pkg-config"""
    return [x for p in packages for x in get_pkg_dirs(p, flag="--cflags-only-I")]


def get_library_dirs(packages: list[str]) -> list[str]:
    """Get library directories from pkg-config"""
    return [x for p in packages for x in get_pkg_dirs(p, flag="--libs-only-L")]


def get_pkg_dirs(package: str, flag: str):
    """Get directories from pkg-config"""
    prefix = flag[-2:]
    try:
        result = subprocess.run(["pkg-config", flag, package], capture_output=True, text=True, check=True)
        return [path.removeprefix(prefix).strip() for path in result.stdout.split() if path.startswith(prefix)]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []  # setuptools will use system defaults


def find_cuda_home() -> str | None:
    """Find CUDA Toolkit installation directory."""
    for path in [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
        "/usr/lib/cuda",
        "/opt/cuda",
    ]:
        if path and os.path.isdir(path):
            return path


class CCUDAExtension(Extension):
    """Like PyTorch's CUDAExtension but for CUDA with pure C bindings instead of C++ and pybind11."""


class BuildExtension(BaseBuildExtension):
    """Custom BuildExtension command that also handles CCUDAExtension."""

    def build_extensions(self):
        for ext in self.extensions:
            if isinstance(ext, CCUDAExtension):
                self._handle_c_cuda_extension(ext)

        super().build_extensions()

    @staticmethod
    def _handle_c_cuda_extension(ext: Extension):
        """Replace .cu sources with pre-built .o files, adjust includes and libs for CUDA."""
        _cuda_home = os.environ["CUDA_HOME"]
        cuda_lib = os.path.join(_cuda_home, "lib64")
        cuda_inc = os.path.join(_cuda_home, "include")
        python_include = sysconfig.get_path("include")

        ext.libraries = (ext.libraries or []) + ["cudart"]
        ext.library_dirs = (ext.library_dirs or []) + [cuda_lib]
        ext.include_dirs = (ext.include_dirs or []) + [cuda_inc, python_include]
        ext.runtime_library_dirs = (ext.runtime_library_dirs or []) + [cuda_lib]

        if not os.path.isfile(nvcc := os.path.join(_cuda_home, "bin", "nvcc")):
            nvcc = "nvcc"

        nvcc_extra_args = []
        if isinstance(ext.extra_compile_args, dict):
            nvcc_extra_args += ext.extra_compile_args.pop("nvcc", None) or []
            ext.extra_compile_args = ext.extra_compile_args.get("cc") or []

        for source in ext.sources.copy():
            if source.endswith(".cu"):
                ext.sources.remove(source)
                compiled = source.removesuffix(".cu") + ".o"
                nvcc_cmd = [nvcc, "-c", source, "-o", compiled]

                for include_dir in ext.include_dirs:
                    nvcc_cmd += ["-I", include_dir]

                subprocess.check_call(nvcc_cmd + nvcc_extra_args)
                ext.extra_objects = (ext.extra_objects or []) + [compiled]


if cuda_available := (cuda_home := find_cuda_home()) is not None:
    os.environ["CUDA_HOME"] = torch.utils.cpp_extension.CUDA_HOME = cuda_home


setup(
    cmdclass={
        "build_ext": BuildExtension,
    },
    ext_modules=[
        x
        for x in [
            get_extension(
                name="my_proj.ext.math",
                path="src/my_proj/ext/_math",
                source_file_types={".c"},
                extra_compile_args=["-std=c11"],
            ),
            get_extension(
                ext_type=CCUDAExtension,
                name="my_proj.ext.linalg",
                path="src/my_proj/ext/_linalg",
                source_file_types={".c", ".cu"},
                include_dirs=[numpy.get_include()],
                extra_compile_args={
                    "cc": ["-std=c11"],
                    "nvcc": ["-std=c++11", "-Xcompiler", "-fPIC"],
                },
            )
            if cuda_available
            else None,
            get_extension(
                ext_type=CUDAExtension if cuda_available else CppExtension,
                name="my_proj.ext.torch_ext",
                path="src/my_proj/ext/_torch_ext",
                source_file_types={".cpp", ".cu"} if cuda_available else {".cpp"},
                extra_compile_args={
                    "cxx": ["-g", "-O0"] if os.environ.get("DEBUG_MODE") else ["-O3"],
                    "nvcc": ["-g", "-G", "-O0"] if os.environ.get("DEBUG_MODE") else ["-O3", "--use_fast_math"],
                },
                runtime_library_dirs=[
                    *torch.utils.cpp_extension.library_paths(),
                ],
                define_macros=[("WITH_CUDA", None)] if cuda_available else [],
            ),
        ]
        if x
    ],
)
