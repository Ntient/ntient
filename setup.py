from setuptools import find_packages, setup

setup(
    name='ntient',
    packages=find_packages(exclude=['tests']),
    version='0.1.1',
    description="Ntient Client Library",
    author="Joel Davenport",
    license="MIT",
    install_requires=['requests'],
    setup_requires=['pytest-runner==5.3.1', 'pytest-mock==3.6.1'],
    tests_require=["pytest==6.2.5", "pytest-mock==3.6.1", "scikit-learn==1.0.2", "tensorflow==2.7.0", "torch==1.10.1"],
    test_suite='tests'
)
