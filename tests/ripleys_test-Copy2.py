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
import logging
import psutil
import os
from datetime import datetime
import time

def setup_logger(output_dir: str, name: str = 'ripley_analysis') -> logging.Logger:
    """Set up logger with file and console handlers."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - Memory: %(memory).2f MB - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(
        os.path.join(log_dir, f'ripley_analysis_{timestamp}.log')
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class MemoryLogFilter(logging.Filter):
    """Add memory usage to log record."""
    def filter(self, record):
        record.memory = get_memory_usage()
        return True

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
        return {
            'gene': gene_name,
            'error': str(e)
        }

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

    # Setup logging
    logger = setup_logger(output_dir)
    logger.addFilter(MemoryLogFilter())
    
    start_time = time.time()
    logger.info(f"Starting Ripley's L analysis for reference gene: {reference_gene}")
    logger.info(f"Initial dataset shape: {adata.shape}")
    
    try:
        # Get spatial coordinates
        coordinates = adata.obsm[spatial_key]
        logger.info(f"Loaded spatial coordinates with shape: {coordinates.shape}")
        
        # Get reference gene expression
        ref_expr = adata[:, reference_gene].X.toarray().flatten()
        ref_high = ref_expr > np.quantile(ref_expr, expression_threshold)
        logger.info(f"Reference gene high expression cells: {np.sum(ref_high)}")
        
        # Prepare support
        hull = ConvexHull(coordinates)
        area = hull.volume
        if max_dist is None:
            max_dist = (area / 2) ** 0.5
        support = np.linspace(0, max_dist, n_steps)
        logger.info(f"Prepared support with max_dist: {max_dist:.2f}")
        
        # Compute null model
        logger.info("Starting null model computation...")
        null_start = time.time()
        rng = default_rng(seed)
        null_distributions = np.empty((n_simulations, len(support)))
        
        for i in range(n_simulations):
            if i % 10 == 0:
                logger.info(f"Computing null distribution {i}/{n_simulations}")
            n_points = max(min_cells, int(np.sum(ref_high) * 0.1))
            random_mask = rng.random(coordinates.shape[0]) < (n_points / coordinates.shape[0])
            random_coords = coordinates[random_mask]
            
            distances = pdist(random_coords)
            n_pairs = (distances < support.reshape(-1, 1)).sum(axis=1)
            intensity = coordinates.shape[0] / area
            k_est = ((n_pairs * 2) / coordinates.shape[0]) / intensity
            null_distributions[i] = np.sqrt(k_est / np.pi)
        
        null_time = time.time() - null_start
        logger.info(f"Completed null model computation in {null_time:.2f} seconds")
        
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
        logger.info("Starting gene data preparation...")
        gene_prep_start = time.time()
        total_genes = len(adata.var_names) - 1  # excluding reference gene
        
        gene_data = []
        for i, gene in enumerate(tqdm(adata.var_names, desc="Preparing genes")):
            if gene == reference_gene:
                continue
            
            if i % 100 == 0:  # Log progress every 100 genes
                current_memory = get_memory_usage()
                logger.info(f"Processing gene {i}/{total_genes} ({gene}). Memory usage: {current_memory/1024:.2f} GB")
            
            gene_data.append((gene, adata[:, gene].X.toarray().flatten()))
        
        gene_prep_time = time.time() - gene_prep_start
        logger.info(f"Completed gene data preparation in {gene_prep_time:.2f} seconds")
        logger.info(f"Prepared analysis for {len(gene_data)} genes")
        
        # Run parallel analysis
        if n_jobs is None:
            n_jobs = mp.cpu_count() - 1
            
        # Log memory state before parallel processing
        current_memory = get_memory_usage()
        logger.info(f"Memory before parallel processing: {current_memory/1024:.2f} GB")
        logger.info(f"Starting parallel analysis using {n_jobs} processes")
        
        # Log size of shared data
        shared_data_size = sum(x.nbytes if hasattr(x, 'nbytes') else 0 
                             for x in shared_data.values() if hasattr(x, '__len__'))
        logger.info(f"Shared data size: {shared_data_size/1024/1024/1024:.2f} GB")
        
        # Prepare arguments list with memory logging
        logger.info("Preparing argument list for parallel processing...")
        args_start_memory = get_memory_usage()
        args = [(g, shared_data) for g in gene_data]
        args_end_memory = get_memory_usage()
        logger.info(f"Arguments list preparation changed memory usage from {args_start_memory/1024:.2f} GB to {args_end_memory/1024:.2f} GB")
        
        parallel_start = time.time()
        try:
            with mp.Pool(n_jobs) as pool:
                worker = partial(_worker_analyze_gene)
                
                # Create a monitoring function to wrap tqdm
                def monitor_progress(iterator):
                    for i, item in enumerate(iterator):
                        if i % 100 == 0:  # Log every 100 genes
                            current_memory = get_memory_usage()
                            logger.info(f"Processed {i} genes. Current memory: {current_memory/1024:.2f} GB")
                        yield item
                
                # Run the parallel processing with monitoring
                results_list = list(tqdm(
                    monitor_progress(pool.imap(worker, args)),
                    total=len(args),
                    desc="Processing genes"
                ))
                
                # Log final memory usage
                final_memory = get_memory_usage()
                logger.info(f"Final memory after parallel processing: {final_memory/1024:.2f} GB")
                
        except Exception as e:
            error_memory = get_memory_usage()
            logger.error(f"Parallel processing failed. Memory at error: {error_memory/1024:.2f} GB")
            logger.error(f"Error during parallel processing: {str(e)}", exc_info=True)
            raise
        
        parallel_time = time.time() - parallel_start
        logger.info(f"Completed parallel analysis in {parallel_time:.2f} seconds")
        
        # Process results
        results_list = [r for r in results_list if r is not None]
        errors = [r for r in results_list if 'error' in r]
        if errors:
            logger.warning(f"Encountered errors in {len(errors)} genes")
            for error in errors[:5]:  # Log first 5 errors
                logger.warning(f"Error in gene {error['gene']}: {error['error']}")
        
        # Create results DataFrame
        valid_results = [r for r in results_list if 'error' not in r]
        df = pd.DataFrame([{
            'gene': r['gene'],
            'max_deviation': r['max_deviation'],
            'min_pvalue': r['min_pvalue']
        } for r in valid_results])
        
        # Sort by max_deviation
        df = df.sort_values('max_deviation', ascending=False)
        
        # Save results
        results_path = os.path.join(output_dir, "ripley_results.csv")
        df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")
        
        # Plot top genes
        logger.info("Generating plots for top genes...")
        top_genes = ['CXCL9'] + df.nlargest(5, 'max_deviation')['gene'].tolist()
        plot_path = os.path.join(output_dir, "ripley_plots.pdf")
        sq.pl.spatial_scatter(
            adata,
            library_id="spatial",
            color=top_genes,
            shape=None,
            size=2,
            save=plot_path
        )
        logger.info(f"Saved plots to {plot_path}")
        
        total_time = time.time() - start_time
        logger.info(f"Analysis completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Clean up logging handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

if __name__ == "__main__":
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
    
    hdir = '/home/workspace/'
    srldir = hdir + 'xen_5k_ibd_ra/objects/'
    output_dir = hdir + 'xen_5k_ibd_ra/'
    
    with open(srldir + 'sdatas.pkl', 'rb') as f:
        sdatas = dill.load(f)
    
    parallel_ripley_test(
        sdatas['ibd'].tables["subsample"],
        reference_gene='GREM1',
        output_dir=output_dir,
        expression_threshold=0.7,
        n_jobs=16,
        n_steps=25
    )