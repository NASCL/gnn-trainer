from setuptools import setup

VERSION = '0.0.1'
DESCRIPTION = 'A Huggingface style trainer for graph neural network models'

setup(
    name='gnn-trainer',
    version=VERSION,
    description=DESCRIPTION,
    author='Joshua Melton',
    author_email='jmelto30@uncc.edu',
    keywords=['pytorch', 'graph machine learning', 'graph representation learning', 'graph neural networks'],
    packages=['gnn-trainer'],
    include_package_data=True,
    install_requires=[
        'numpy>=1.16.0',
        'requests',
        'scikit-learn>=0.20.0',
        'tqdm',
        'torch>=1.6.0',
    ],
)
