import os

from pydefect_2d import __version__
from setuptools import setup, find_packages

cmdclass = {}
ext_modules = []

module_dir = os.path.dirname(os.path.abspath(__file__))
reqs_raw = open(os.path.join(module_dir, "requirements.txt")).read()
reqs_list = [r.replace("==", "~=") for r in reqs_raw.split("\n")]

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setup(
    name='pydefect_2d',
    version=__version__,
    author='Yu Kumagai',
    author_email='yukumagai@tohoku.ac.jp',
    url='https://github.com/kumagai-group/pydefect_2d',
    packages=find_packages(),
    license='MIT license',
    description="Package for correcting defect formation energies and "
                "eigenvalues in two-dimensional materials",
    classifiers=[
        'Programming Language :: Python :: 3.10',
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=reqs_list,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pydefect_2d = pydefect_2d.cli.main:main',
            'pydefect_2d_util = pydefect_2d.cli.main_util:main',
            'pydefect_2d_plot = pydefect_2d.cli.main_plot_json:main',
        ]
    }
)
