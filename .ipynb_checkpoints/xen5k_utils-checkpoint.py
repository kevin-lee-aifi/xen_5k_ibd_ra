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


def read_from_zip(zip_file):
    with tempdir() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for root, dir, files in os.walk(temp_dir):
            if 'output-XET' in root:
                path = root
                break
                
        sdata = xenium(path)
        
        return(sdata)



def process_adata(adata):
    sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 150), inplace=True)
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    sq.gr.centrality_scores(adata, cluster_key="leiden")
    sq.gr.nhood_enrichment(adata, cluster_key="leiden")

    return(adata)