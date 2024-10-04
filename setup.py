# in terminal, run: python setup.py install

from setuptools import setup, find_packages

setup(
    name='ML',  # Name of your project
    description='Machine learning code for Lu, Jia, et al. Decoding pattern formation rules by integrating mechanistic modeling and deep learning." bioRxiv (2024)',
    author='Jia Lu',
    packages=find_packages(),  
    install_requires=['numpy', 
                      'pandas', 
                      'sklearn', 
                      'torch', 
                      'tqdm', 
                      'matplotlib'],
    python_requires='>=3.8',
)
