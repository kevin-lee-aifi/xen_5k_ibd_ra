from typing import Dict, Union, Optional, Sequence
import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.random import default_rng
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import multiprocessing as mp
from scanpy import logging as logg
import squidpy as sq

def _worker_analyze_gene(args: tuple) -> dict:
    gene_data, shared_data = args
    gene_name, gene_expr = gene_data
    
    try:
        # Unpack shared data
        coordinates = shared_data['coordinates']
        ref_high = shared_data['ref_high']
        support = shared_data['support']
        null_distributions = shared_data['null_distributions']
        expression_threshold = shared_data['expression_threshold']
        min_cells = shared_data['min_cells']
        N = shared_data['N']
        area = shared_data['area']

        # Process gene
        test_high = gene_expr > np.quantile(gene_expr, expression_threshold)
        joint_coords = coordinates[ref_high & test_high]
        
        if joint_coords.shape[0] < min_cells:
            return None
            
        # Calculate L function
        distances = pdist(joint_coords)
        n_pairs_less_than_d = (distances < support.reshape(-1, 1)).sum(axis=1)
        intensity = N / area
        k_estimate = ((n_pairs_less_than_d * 2) / N) / intensity
        l_stat = np.sqrt(k_estimate / np.pi)
        
        # Calculate statistics
        max_deviation = np.max(np.abs(l_stat - np.mean(null_distributions, axis=0)))
        pvalues = np.array([
            (np.sum(null_distributions[:, d] >= l_stat[d]) + 1) / (len(null_distributions) + 1)
            for d in range(len(support))
        ])
        
        return {
            'gene': gene_name,
            'l_stat': l_stat,
            'max_deviation': max_deviation,
            'min_pvalue': np.min(pvalues),
            'pvalues': pvalues
        }
    except Exception as e:
        print(f"Error processing gene {gene_name}: {str(e)}")
        return None

def parallel_ripley_test(
    adata: AnnData,
    reference_gene: str,
    output_dir: str,
    spatial_key: str = 'spatial',
    expression_threshold: float = 0,
    n_simulations: int = 100,
    max_dist: Optional[float] = None,
    n_steps: int = 50,
    min_cells: int = 10,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None
) -> None:

    start = logg.info(f"Running Ripley's L analysis for {reference_gene}")
    
    # Get spatial coordinates
    coordinates = adata.obsm[spatial_key]
    
    # Get reference gene expression
    ref_expr = adata[:, reference_gene].X.toarray().flatten()
    ref_high = ref_expr > np.quantile(ref_expr, expression_threshold)
    
    # Prepare support
    hull = ConvexHull(coordinates)
    area = hull.volume
    if max_dist is None:
        max_dist = (area / 2) ** 0.5
    support = np.linspace(0, max_dist, n_steps)
    
    # Compute null model
    print("Computing null model...")
    rng = default_rng(seed)
    null_distributions = np.empty((n_simulations, len(support)))
    
    for i in range(n_simulations):
        n_points = max(min_cells, int(np.sum(ref_high) * 0.1))
        random_mask = rng.random(coordinates.shape[0]) < (n_points / coordinates.shape[0])
        random_coords = coordinates[random_mask]
        
        distances = pdist(random_coords)
        n_pairs = (distances < support.reshape(-1, 1)).sum(axis=1)
        intensity = coordinates.shape[0] / area
        k_est = ((n_pairs * 2) / coordinates.shape[0]) / intensity
        null_distributions[i] = np.sqrt(k_est / np.pi)
    
    # Prepare shared data
    shared_data = {
        'coordinates': coordinates,
        'ref_high': ref_high,
        'support': support,
        'null_distributions': null_distributions,
        'expression_threshold': expression_threshold,
        'min_cells': min_cells,
        'N': coordinates.shape[0],
        'area': area
    }
    
    # Prepare gene data
    gene_data = [
        (g, adata[:, g].X.toarray().flatten()) 
        for g in adata.var_names if g != reference_gene
    ]
    
    # Run parallel analysis
    if n_jobs is None:
        n_jobs = mp.cpu_count() - 1
    print(f"Analyzing genes using {n_jobs} processes...")
    
    with mp.Pool(n_jobs) as pool:
        worker = partial(_worker_analyze_gene)
        args = [(g, shared_data) for g in gene_data]
        results_list = list(tqdm(
            pool.imap(worker, args),
            total=len(args)
        ))
    
    # Process results
    results_list = [r for r in results_list if r is not None]
    
    # Create results DataFrame
    df = pd.DataFrame([{
        'gene': r['gene'],
        'max_deviation': r['max_deviation'],
        'min_pvalue': r['min_pvalue']
    } for r in results_list])
    
    # Sort by max_deviation
    df = df.sort_values('max_deviation', ascending=False)
    
    # Save results
    df.to_csv(f"{output_dir}/ripley_results.csv", index=False)
    print(f"Results saved to {output_dir}/ripley_results.csv")
    
    # Plot top genes
    top_genes = ['CXCL9'] + df.nlargest(5, 'max_deviation')['gene'].tolist()
    sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        color=top_genes,
        shape=None,
        size=2,
        save=f"{output_dir}/ripley_plots.pdf"
    )
    
    logg.info("Finish", time=start)

if __name__ == "__main__":
    # Example usage
    from xen5k_utils import *
    import dill
    
    hdir = '/home/workspace/'
    srldir = hdir + 'xen_5k_ibd_ra/xen5k_objects/'
    output_dir = hdir + 'xen_5k_ibd_ra/'
    
    with open(srldir + 'sdatas.pkl', 'rb') as f:
        sdatas = dill.load(f)
    
    parallel_ripley_test(
        sdatas['ibd'].tables["subsample"],
        reference_gene='GREM1',
        output_dir=output_dir,
        expression_threshold=0.7, # Can I test a range from 0 to 1.0 in increments of 0.2?
        n_jobs=16
    )