#!/usr/bin/bash

ENV="$HOME/environment/xen5k_env"

conda create -y -p $ENV -c conda-forge \
    python=3.9 \
    ipykernel \
    h5py \
    scanpy \
    numpy==1.24.4 \
    pandas \
    anndata \
    matplotlib \
    seaborn \
    squidpy \
    spatialdata \
    spatialdata-io \
    spatialdata-plot \
    dill \
    
conda activate $ENV

pip install igraph \
    git+https://github.com/complextissue/pyclustree.git@main \
    git+https://github.com/StatBiomed/SpatialDM \
    psutil

python -m ipykernel install --user --name xen5k_env --display-name "Python (xen5k_env)"

cd "$HOME/xen_5k_ibd_ra"