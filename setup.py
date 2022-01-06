from setuptools import find_packages, setup

setup(
    name='ntient',
    packages=find_packages(include=['ntient']),
    version='0.1.0',
    description="Ntient Client Library",
    author="Joel Davenport",
    license="MIT",
    install_requires=['requests'],
    setup_requires=['pytest-runner==5.3.1', 'pytest-mock==3.6.1'],
    tests_requires=['pytest==6.2.5', 'pytest-mock==3.6.1'],
    test_suite='tests'
)
