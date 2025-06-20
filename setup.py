from setuptools import setup, find_packages

setup(
    name='sdea',
    version='0.1.0',
    description='Shift Distribution Estimation and Alignment module (SDEA) for ML',
    author='Nitipon Pongphaw',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10',
    ],
    python_requires='>=3.6',
)
