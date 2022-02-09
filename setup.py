from setuptools import find_packages, setup

long_description = open("README.md").read()

setup(
    name='ntient',
    packages=find_packages(exclude=['tests']),
    version='0.1.6',
    description="Ntient Client Library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Joel Davenport",
    license="MIT",
    install_requires=['requests'],
    setup_requires=['pytest-runner==5.3.1', 'pytest-mock==3.6.1'],
    tests_require=["pytest==6.2.5", "pytest-mock==3.6.1", "scikit-learn==1.0.2", "tensorflow==2.7.0", "torch==1.10.1"],
    test_suite='tests'
)
