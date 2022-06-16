from setuptools import setup

DISTNAME = 'gnn-trainer'
DESCRIPTION = 'A HuggingFace style trainer for graph neural network models'
MAINTAINER = 'Joshua Melton'
MAINTAINER_EMAIL = 'jmelto30@uncc.edu'
URL = 'https://github.com/NASCL/gnn-trainer'
LICENSE = 'MIT'

import gnn_trainer

VERSION = gnn_trainer.__version__

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    license=LICENSE,
    keywords=['pytorch', 'graph machine learning', 'graph representation learning', 'graph neural networks'],
    packages=['gnn_trainer'],
    include_package_data=True,
    install_requires=[
        'numpy>=1.16.0',
        'requests',
        'scikit-learn>=0.20.0',
        'tqdm',
        'torch>=1.6.0',
    ],
)
