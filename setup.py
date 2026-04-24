from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "models.vebpr_engine",
        sources=["models/vebpr_engine.pyx"],
        language="c++",
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name='ShopeeRanking',
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[numpy.get_include()],
)