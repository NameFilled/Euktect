"""
Automated MAG Refinement System with Precision Control

This script implements a systematic approach to refine Metagenome-Assembled Genomes (MAGs) 
using machine learning predictions and quality assessment tools. It features:
- Precision-controlled calculations using Decimal module
- Binary search algorithm for optimal threshold finding
- Integration with EukCC quality assessment tool
- Caching mechanism for efficient computation
- Comprehensive logging and result tracking
"""

import os
import shutil
import pandas as pd
from pyfaidx import Fasta
import subprocess
import logging
import argparse
from decimal import Decimal, getcontext

# Global precision configuration for decimal operations
DEFAULT_PRECISION = 16
getcontext().prec = 25  # Higher precision for calculations than storage

def decimal_quantize(value, precision=DEFAULT_PRECISION):
    """Quantize Decimal values to ensure consistent precision across operations.
    
    Args:
        value: Input value (int/float/str/Decimal)
        precision: Number of decimal places to keep
        
    Returns:
        Quantized Decimal value
    """
    return Decimal(str(value)).quantize(Decimal(f'1.{"0"*precision}'))

def refine_sequences(threshold, input_fasta_path, refined_dir, df):
    """Filter FASTA sequences based on prediction probabilities.
    
    Args:
        threshold: Probability cutoff for sequence selection
        input_fasta_path: Path to input FASTA file
        refined_dir: Output directory for filtered sequences
        df: DataFrame containing sequence IDs and probabilities
        
    Returns:
        Path to generated FASTA file or None if failed
    """
    try:
        decimal_threshold = decimal_quantize(threshold)
        
        # Select sequences based on threshold
        if decimal_threshold == Decimal(0):
            sequence_ids = df['sequence_id'].tolist()
        else:
            filtered_df = df[df['label_0_probability'] > decimal_threshold]
            sequence_ids = filtered_df['sequence_id'].tolist()
        
        # Extract sequences from FASTA
        fasta = Fasta(input_fasta_path)
        sequences = []
        for seq_id in sequence_ids:
            try:
                seq = str(fasta[seq_id])
                sequences.append(f">{seq_id}\n{seq}\n")
            except KeyError:
                logging.warning(f"Sequence {seq_id} not found in {input_fasta_path}")
        
        if not sequences:
            return None
        
        # Write output file
        os.makedirs(refined_dir, exist_ok=True)
        output_path = os.path.join(refined_dir, f"refined_{decimal_threshold:.16f}.fa")
        with open(output_path, 'w') as f:
            f.writelines(sequences)
        return output_path
    except Exception as e:
        logging.error(f"Refine sequences failed: {str(e)}")
        return None

def run_eukcc(refined_file, threshold, eukcc_result_dir, eukcc_db):
    """Execute EukCC quality assessment tool on refined sequences.
    
    Args:
        refined_file: Path to filtered FASTA file
        threshold: Current probability threshold being tested
        eukcc_result_dir: Directory for EukCC output
        eukcc_db: Path to EukCC database
        
    Returns:
        Tuple of (completeness, contamination) as Decimals or (None, None) if failed
    """
    try:
        output_dir = os.path.join(eukcc_result_dir, f"refined_{decimal_quantize(threshold):.16f}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Build EukCC command
        cmd = f"eukcc single --out {output_dir} --threads 64 {refined_file} --db {eukcc_db}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"EukCC error (threshold={threshold}): {result.stderr}")
            return None, None
        
        # Parse results
        result_file = os.path.join(output_dir, 'eukcc.csv')
        if os.path.exists(result_file):
            df = pd.read_csv(result_file, sep='\t')
            comp = decimal_quantize(df['completeness'].iloc[0]/100, 4)
            ctm = decimal_quantize(df['contamination'].iloc[0]/100, 4)
            logging.info(f"Threshold: {decimal_quantize(threshold)} → Comp: {comp}, Contam: {ctm}")
            return comp, ctm
        return None, None
    except Exception as e:
        logging.error(f"Run EukCC failed: {str(e)}")
        return None, None

def load_result_cache(report_path):
    """Load historical results from cache file.
    
    Args:
        report_path: Path to CSV cache file
        
    Returns:
        Dictionary of cached results {threshold: (completeness, contamination)}
    """
    cache = {}
    if os.path.exists(report_path):
        try:
            df = pd.read_csv(report_path)
            for _, row in df.iterrows():
                if pd.notna(row['completeness']) and pd.notna(row['contamination']):
                    threshold = decimal_quantize(row['threshold'])
                    comp = decimal_quantize(row['completeness'], 4)
                    ctm = decimal_quantize(row['contamination'], 4)
                    cache[threshold] = (comp, ctm)
        except Exception as e:
            logging.warning(f"Load cache failed: {str(e)}")
    return cache

def binary_search(input_fasta_path, df, quality_standard, workdir, eukcc_db, result_cache=None):
    """Enhanced binary search algorithm for finding optimal probability threshold.
    
    Args:
        input_fasta_path: Input MAG FASTA file path
        df: DataFrame with sequence probabilities
        quality_standard: Target quality standard ('HQ' or 'MQ')
        workdir: Working directory for outputs
        eukcc_db: Path to EukCC database
        result_cache: Pre-loaded results cache
        
    Returns:
        Tuple of (best_result_dict, updated_cache)
    """
    base_name = os.path.splitext(os.path.basename(input_fasta_path))[0]
    logging.info(f"Starting {quality_standard} search for {base_name}")
    
    report_path = os.path.join(workdir, 'report', f'{base_name}_threshold_completeness_contamination.csv')
    
    # Initialize cache
    cache = load_result_cache(report_path)
    if result_cache:
        cache.update(result_cache)
        
    # Quality standards configuration
    standards = {
        'HQ': {'comp': Decimal('0.9'), 'contam': Decimal('0.05')},
        'MQ': {'comp': Decimal('0.5'), 'contam': Decimal('0.1')}
    }
    target = standards[quality_standard]
    
    # Prepare directories
    refined_dir = os.path.join(workdir, 'refined', base_name)
    eukcc_dir = os.path.join(workdir, 'eukcc_refined_result', base_name)
    os.makedirs(refined_dir, exist_ok=True)
    os.makedirs(eukcc_dir, exist_ok=True)
    
    # Binary search parameters
    low, high = Decimal(0), Decimal(1)
    best = {'threshold': None, 'comp': Decimal(0), 'contam': Decimal(1)}
    precision_limit = Decimal('1e-16')
    
    while True:
        mid = decimal_quantize((low + high) / 2)
        logging.info(f"Testing threshold={mid:.6f} for {quality_standard} standards...")
        
        # Check precision termination condition
        if (high - low) <= precision_limit:
            logging.info("Reached precision limit of 1e-16. Stopping search.")
            break
        
        # Check cache for existing results
        if mid in cache:
            comp, contam = cache[mid]
            logging.info(f"Using cached result for threshold={mid:.6f}: Comp={comp:.4f}, Contam={contam:.4f}")
        else:
            # Process new threshold
            refined_file = refine_sequences(mid, input_fasta_path, refined_dir, df)
            comp, contam = run_eukcc(refined_file, mid, eukcc_dir, eukcc_db)
            cache[mid] = (comp, contam)
            
            # Record results
            with open(report_path, 'a') as f:
                comp_str = f"{comp:.4f}" if comp else ""
                ctm_str = f"{contam:.4f}" if contam else ""
                f.write(f"{mid:.16f},{comp_str},{ctm_str}\n")
        
        # Handle invalid results
        if comp is None or contam is None:
            high = mid
            continue
        
        # Update best candidate
        if (comp >= target['comp'] and contam <= target['contam'] and 
            (best['threshold'] is None or mid < best['threshold'])):
            best.update(threshold=mid, comp=comp, contam=contam)
            logging.info(f"New best threshold found: {mid:.6f} (Comp: {comp:.4f}, Contam: {contam:.4f})")
        
        # Adjust search boundaries
        if contam > target['contam']:
            if comp >= target['comp']:
                low = mid
            else:
                if quality_standard == "MQ":
                    logging.info("MQ requirements cannot be met")
                break
        else:
            if comp >= target['comp']:
                high = mid
            else:
                high = mid
    
    # Save final results
    if best['threshold'] is not None:
        result_dir = os.path.join(workdir, 'refined_result', base_name)
        os.makedirs(result_dir, exist_ok=True)
        src_file = refine_sequences(best['threshold'], input_fasta_path, refined_dir, df)
        if src_file:
            dest_name = f"refined_{base_name}_{quality_standard}_thd{best['threshold']:.16f}_cpt{best['comp']:.4f}_ctm{best['contam']:.4f}.fa"
            shutil.copy(src_file, os.path.join(result_dir, dest_name))
            logging.info(f"Found {quality_standard} solution: {dest_name}")
    
    return best, cache

def main():
    """Main execution flow handling command line arguments and workflow orchestration."""
    parser = argparse.ArgumentParser(description="Refine MAG using predicted probabilities")
    parser.add_argument("--fasta", type=str, required=True, help="Input MAG FASTA file")
    parser.add_argument("--prob_file", type=str, required=True, help="CSV file with sequence probabilities")
    parser.add_argument("--workdir", type=str, required=True, help="Output working directory")
    parser.add_argument("--eukcc_db", type=str, required=True, help="EukCC database path")
    parser.add_argument("--input_MAG_quality", type=str, choices=['LQ', 'MQ'], 
                      default='LQ', help="Initial MAG quality level")
    args = parser.parse_args()
    
    # Initialize workspace
    for subdir in ['report', 'log', 'refined_result', 'refined', 'eukcc_refined_result']:
        os.makedirs(os.path.join(args.workdir, subdir), exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(args.workdir, 'log', f"{os.path.basename(args.fasta)}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    try:
        # Load prediction data
        df = pd.read_csv(args.prob_file)
        df['label_0_probability'] = df['label_0_probability'].apply(decimal_quantize)
        
        # Initialize report file
        base_name = os.path.splitext(os.path.basename(args.fasta))[0]
        report_path = os.path.join(args.workdir, 'report', f'{base_name}_threshold_completeness_contamination.csv')
        if not os.path.exists(report_path):
            with open(report_path, 'w') as f:
                f.write("threshold,completeness,contamination\n")
                
        # Execute refinement based on input quality
        if args.input_MAG_quality == 'LQ':
            # Attempt HQ refinement first
            hq_result, cache = binary_search(args.fasta, df, 'HQ', args.workdir, args.eukcc_db)
            if hq_result['threshold'] is not None:
                return
            
            # Fallback to MQ refinement
            logging.info("No HQ solution, starting MQ search...")
            mq_result, _ = binary_search(args.fasta, df, 'MQ', args.workdir, args.eukcc_db, cache)
        
        elif args.input_MAG_quality == 'MQ':
            # Direct HQ refinement attempt
            logging.info("Input is MQ quality, only attempting HQ refinement")
            hq_result, _ = binary_search(args.fasta, df, 'HQ', args.workdir, args.eukcc_db)
            
    except Exception as e:
        logging.error(f"Error processing {args.fasta}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()