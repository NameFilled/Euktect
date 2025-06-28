## Euktect: A Deep Learning Model for Alignment-Free Taxonomic Classification of Eukaryotic DNA Sequences in Metagenomes

Euktect is a deep learning model built upon the Hyena-DNA architecture, designed for alignment-free taxonomic classification of DNA sequences. With Euktect, you can classify DNA sequences across various taxonomic levels without the need for sequence alignment.

------

## Setup

1. **Clone the Repository** (including submodules):

   ```bash
   git clone --recurse-submodules git@github.com:NameFilled/Euktect.git
   ```

2. **Create a Conda Environment** (recommended):

   ```bash
   conda create -n Euktect python=3.8
   conda activate Euktect
   ```

3. **Install PyTorch and CUDA** (ensure your CUDA driver is >= 11.7):

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
   ```

4. **Install Python Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

> **Note:** If you encounter issues with environment compatibility, refer to the [Hyena-DNA repository](https://github.com/HazyResearch/hyena-dna) for detailed setup instructions, or contact [bufifhei@foxmail.com](mailto:bufifhei@foxmail.com).

------

## Model Usage

Before running predictions, download the model checkpoint (`.ckpt`) and configuration file (`.yaml`) from the [Releases](https://github.com/NameFilled/Euktect/releases) page.

```bash
python predict.py \
  --input <path_to_fasta_file> \
  --ckpt <path_to_model_ckpt> \
  --cfg <path_to_model_cfg> \
  --output <path_to_report_output>
```

- **Optional**: `--max_chunk` to set the maximum number of sequence chunks (default: 10).

The prediction results will be saved as a CSV file with the following columns:

```
sequence_id,prob_label_0,...,prob_label_n,predict_label
```

The label mappings for each model are defined below.

### Superkingdom Model

| Label | Class     |
| ----- | --------- |
| 0     | Eukaryote |
| 1     | Bacteria  |
| 2     | Archaea   |
| 3     | Virus     |

### Fungal Phylum Multi-class Model

| Label | Phylum             |
| ----- | ------------------ |
| 0     | Ascomycota         |
| 1     | Basidiomycota      |
| 2     | Blastocladiomycota |
| 3     | Chytridiomycota    |
| 4     | Cryptomycota       |
| 5     | Microsporidia      |
| 6     | Mucoromycota       |
| 7     | Olpidiomycota      |
| 8     | Sanchytriomycota   |
| 9     | Zoopagomycota      |

### Candida Genus Multi-level Model

Each level uses a binary classifier with the following label encodings:

- **Kingdom**: Fungi (0), Other Eukaryote (1)
- **Phylum**: Ascomycota (0), Other Fungi (1)
- **Class**: Pichiomycetes (0), Other Ascomycota (1)
- **Order**: Saccharomycetales (0), Other Pichiomycetes (1)
- **Family**: Debaryomycetaceae (0), Other Saccharomycetales (1)
- **Genus**: Candida (0), Other Debaryomycetaceae (1)

------

## MAG Refinement Workflow

To refine eukaryotic metagenome-assembled genomes (MAGs), combine Euktect’s superkingdom predictions with a MAG quality tool such as EukCC.

1. **Predict MAG Sequences**:

   ```bash
   python predict.py \
     --input <path_to_MAG_fasta> \
     --ckpt <superkingdom_model_ckpt> \
     --cfg <superkingdom_model_cfg> \
     --output <path_to_report_output>
   ```

2. **Install EukCC and Dependencies**:

   ```bash
   pip install eukcc
   ```

   Additional dependencies:

   - metaeuk==4.a0f584d
   - pplacer
   - epa-ng==0.3.8
   - hmmer==3.3
   - minimap2
   - bwa
   - samtools

   Download the EukCC database as described in the [EukCC repository](https://github.com/EBI-Metagenomics/EukCC).

3. **Run Refinement**:

   ```bash
   python refine.py \
     --fasta <path_to_MAG_fasta> \
     --prob_file <path_to_report_output> \
     --workdir <path_to_workdir> \
     --eukcc_db <path_to_eukcc_db>
   ```
    - **Optional**: `--input_MAG_quality` to set the initial MAG quality level (default: LQ).
   If set to LQ, the refinement will sequentially search for HQ and MQ bins;
   if set to MQ, it will only search for HQ bins.

4. **Review Output in `workdir/`**:

   ```plaintext
   workdir/
   ├── report/                 # Quality assessment reports
   ├── log/                    # Processing logs
   ├── refined_result/         # Final refined MAG sequences
   ├── refined/                # Intermediate filtered sequences
   └── eukcc_refined_result/   # Original EukCC output
   ```

Refined MAG sequences will be available in the `workdir/refined_result/` directory.

## Citation

If you use Euktect in your research, please cite the following paper:

> Peng Y, Ji B, Wang Y, et al. Euktect: Enhanced Eukaryotic Sequence Detection and Classification in Metagenomes via the DNA Language Model[J]. bioRxiv, 2025: 2025.06. 19.660294.
