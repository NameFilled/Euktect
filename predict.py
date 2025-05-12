"""
DNA Sequence Prediction Script with Euktect Model
Version: 1.0
Features:
- Processes FASTA input files
- Supports configurable model parameters
- Enables GPU acceleration
- Implements sequence chunking for long sequences
- Generates CSV reports with prediction probabilities
"""

import torch
import argparse
import os
import sys
import yaml
import csv
import numpy as np
from pyfaidx import Fasta  # For efficient FASTA file parsing
from tqdm import tqdm  # For progress bar visualization

# Add hyena-dna module to Python path
sys.path.append(os.path.abspath("hyena-dna"))
from src.models.sequence.long_conv_lm import DNAEmbeddingModel
from src.tasks.decoders import SequenceDecoder
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

class ModelLoader:
    """Handles model configuration, loading, and inference operations.
    
    Attributes:
        cfg_path (str): Path to model configuration YAML file
        ckp_path (str): Path to model checkpoint file
        device (str): Computation device ('cuda' or 'cpu')
        max_chunk (int): Maximum number of sequence chunks for processing
        tokenizer: DNA sequence tokenizer
        backbone: HyenaDNA model architecture
        decoder: Prediction head for classification
    """

    def __init__(self, cfg_path, ckp_path, device, max_chunk=10):
        """Initialize ModelLoader with configuration and hardware setup."""
        self.cfg_path = cfg_path
        self.ckp_path = ckp_path
        self.device = device
        self.max_chunk = max_chunk  # Maximum allowed sequence chunks
        self.tokenizer = None
        self.backbone = None
        self.decoder = None
        self.load_config()
        self.load_model()

    def load_config(self):
        """Load model configuration from YAML file and initialize tokenizer."""
        with open(self.cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        self.model_cfg = cfg['model']
        # Get maximum sequence length or default to 1000
        self.max_seq_len = self.model_cfg['max_seq_len']
        # Initialize DNA character tokenizer
        self.tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # DNA bases + unknown
            model_max_length=self.max_seq_len + 2,  # Add buffer for tokens
            add_special_tokens=False  # No special tokens for DNA sequences
        )

    def load_model(self):
        """Load model architecture and weights from checkpoint."""
        # Initialize model components
        self.backbone = DNAEmbeddingModel(**self.model_cfg, return_hidden_state=True)
        self.decoder = SequenceDecoder(
            self.model_cfg['d_model'],  # Input dimension
            d_output=self.model_cfg["d_output"],  # Output classes
            l_output=0,  # Output length (0 for classification)
            mode='last'  # Use last position for classification
        )

        # Load checkpoint and process state dict
        state_dict = torch.load(self.ckp_path, map_location='cpu')
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )
        model_state_dict = state_dict["state_dict"]

        # Extract decoder weights
        decoder_state_dict = {
            'output_transform.weight': model_state_dict.pop('decoder.0.output_transform.weight'),
            'output_transform.bias': model_state_dict.pop('decoder.0.output_transform.bias')
        }

        # Load weights into model components
        self.decoder.load_state_dict(decoder_state_dict, strict=True)
        self.backbone.load_state_dict(model_state_dict, strict=False)

        # Move models to target device
        self.backbone = self.backbone.to(self.device)
        self.decoder = self.decoder.to(self.device)

    def predict(self, seq_r):
        """Run prediction on a DNA sequence.
        
        Args:
            seq_r (str): Raw DNA sequence string
            
        Returns:
            np.ndarray: Class probability distribution
        """
        # Sequence chunking logic for long sequences
        if len(seq_r) <= self.max_seq_len:
            seqs = [seq_r]
        else:
            seqs = []
            num_subseq = len(seq_r) // self.max_seq_len
            
            if num_subseq < self.max_chunk:
                # Split into full-length chunks
                for x in range(1, num_subseq + 1):
                    start = (x - 1) * self.max_seq_len
                    end = x * self.max_seq_len
                    seqs.append(seq_r[start:end])
                # Add remaining sequence
                if seq_r[num_subseq * self.max_seq_len:]:
                    seqs.append(seq_r[-self.max_seq_len:])
            else:
                # Take first max_chunk chunks
                for x in range(1, self.max_chunk + 1):
                    start = (x - 1) * self.max_seq_len
                    end = x * self.max_seq_len
                    seqs.append(seq_r[start:end])

        sub_seq_preds = []
        for seq in seqs:
            # Pad short sequences with N's
            seq = seq.ljust(self.max_seq_len, "N")
            
            with torch.no_grad():
                # Tokenize and convert to tensor
                tokenized = self.tokenizer.encode(seq)
                inputs = torch.tensor([tokenized]).to(self.device)
                
                # Model inference
                embedding, _ = self.backbone(inputs)
                pred = self.decoder(embedding)
                sub_seq_preds.append(pred)
        
        # Aggregate predictions
        stacked_preds = torch.stack(sub_seq_preds)
        seq_pred = torch.mean(stacked_preds, dim=0)  # Average predictions
        probabilities = torch.softmax(seq_pred, dim=-1).cpu().numpy().flatten()
        return probabilities

    def save_results_to_csv(self, sequence_id, probabilities, output_file):
        """Save prediction results to CSV file.
        
        Args:
            sequence_id (str): Unique sequence identifier
            probabilities (np.ndarray): Class probability vector
            output_file (str): Output CSV file path
        """
        # Generate CSV headers
        label_headers = [f"label_{i}_probability" for i in range(len(probabilities))]
        header = ["sequence_id"] + label_headers + ["predict_label"]
        
        # Create data row
        predict_label = np.argmax(probabilities)
        row = [sequence_id] + probabilities.tolist() + [predict_label]

        # Write header if file doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        # Append new row
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def main():
    """Main execution function for command-line interface."""
    # Configure command-line arguments
    parser = argparse.ArgumentParser(
        description="DNA Sequence Classification with HyenaDNA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, 
                      help="Input FASTA file path")
    parser.add_argument("--ckpt", required=True,
                      help="Model checkpoint file path")
    parser.add_argument("--cfg", required=True,
                      help="Model configuration YAML file")
    parser.add_argument("--output", required=True,
                      help="Output CSV report path")
    parser.add_argument("--max_chunk", type=int, default=10,
                      help="Maximum sequence chunks for long reads")
    args = parser.parse_args()

    # Clear existing output file
    if os.path.exists(args.output):
        os.remove(args.output)

    # Initialize hardware configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and configurations
    model_loader = ModelLoader(
        cfg_path=args.cfg,
        ckp_path=args.ckpt,
        device=device,
        max_chunk=args.max_chunk
    )

    # Process input sequences
    fasta = Fasta(args.input)
    for seq_id in tqdm(fasta.keys(), desc="Processing sequences"):
        sequence = str(fasta[seq_id])
        probs = model_loader.predict(sequence)
        model_loader.save_results_to_csv(seq_id, probs, args.output)

    print(f"Prediction report generated at: {args.output}")

if __name__ == "__main__":
    main()