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

with open(srldir + 'sdatas.pkl', 'rb') as f:
    sdatas = dill.load(f)

def sdm_processing()

adata = sdatas['ibd'].tables['table']
print(adata)

adata.raw = adata.copy()

sdm.weight_matrix(adata, l=75, cutoff=0.2, single_cell=False)
sdm.extract_lr(adata, 'human', min_cell=3)
sdm.spatialdm_global(adata, 1000, specified_ind=None, method='z-score', nproc=8)
sdm.sig_pairs(adata, method='z-score', fdr=True, threshold=0.1)
sdm.spatialdm_local(adata, n_perm=1000, method='z-score', specified_ind=None, nproc=8)
sdm.sig_spots(adata, method='z-score', fdr=False, threshold=0.1)

with open(srldir + 'sdm_test_sdata.pkl', 'wb') as f:
    dill.dump(sdatas['ibd'], f)

print(adata)