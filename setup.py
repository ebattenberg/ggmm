from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='ggmm',
    version='0.1.0',
    description='Python module to train GMMs using CUDA',
    long_description=long_description,
    url='https://github.com/ebattenberg/ggmm',
    author='Eric Battenberg',
    author_email='ebattenberg@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering'
    ],
    keywords='GMM CUDA',
    packages=find_packages(exclude=['tests']),
    install_requires=['CUDAMat'],
    python_requires='==2.*'
)
