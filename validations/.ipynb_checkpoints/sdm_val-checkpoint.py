import numpy as np
import pandas as pd
import scanpy as sc
import h5py as h5
import anndata as ad
import matplotlib.pyplot as plt
import squidpy as sq
import seaborn as sns
from joblib import dump, load
import os
from tempfile import TemporaryDirectory as tempdir
import spatialdata as sd
from pyclustree import clustree
from spatialdata_io import xenium
import spatialdata_plot
import zipfile
import dill
import spatialdm as sdm

hdir = '/home/workspace/'
srldir = hdir + 'xen_5k_ibd_ra/xen5k_objects/'

print("Loading sdatas")
with open(srldir + 'sdatas.pkl', 'rb') as f:
    sdatas = dill.load(f)
    
adata = sdatas['ibd'].tables['table']
print(adata)

# Store raw data before any processing
adata.raw = adata.copy()

print("Running weight_matrix")
sdm.weight_matrix(adata, l=75, cutoff=0.2, single_cell=False)

print("Running extract_lr")
sdm.extract_lr(adata, 'human', min_cell=3)

print("Running spatialdm_global")
sdm.spatialdm_global(adata, 1000, specified_ind=None, method='z-score', nproc=8)

print("Running sig_pairs")
sdm.sig_pairs(adata, method='z-score', fdr=True, threshold=0.1)

print("Running spatialdm_local")
sdm.spatialdm_local(adata, n_perm=1000, method='z-score', specified_ind=None, nproc=8)

print("Running sig_spots")
sdm.sig_spots(adata, method='z-score', fdr=False, threshold=0.1)

with open(srldir + 'sdm_test_sdata.pkl', 'wb') as f:
    dill.dump(sdatas['ibd'], f)

print(adata)