from distutils.core import setup
from distutils.extension import Extension

setup(name="PackageName",
    ext_modules=[
        Extension("hello", ["scan_matcher.cpp"],
        libraries = ["boost_python"])
    ])