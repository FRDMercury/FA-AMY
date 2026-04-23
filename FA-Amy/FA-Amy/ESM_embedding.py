import os
import numpy as np
import torch
from tqdm import tqdm
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig, LogitsOutput

# ======================================================
# CONFIGURATION AREA - PLEASE MODIFY THE PATHS BELOW
# ======================================================
# 1. Path to your input FASTA/TXT file
INPUT_FASTA_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Benchmark_dataset", "Neg-Train.txt")

# 2. Path to save the generated embedding (.npy)
OUTPUT_NPY_PATH = "esmc_neg_test1.npy"

# 3. Model & Sequence Settings
MODEL_NAME = "esmc_600m"
MAX_LENGTH = 900  # Expected maximum sequence length for padding/truncating
# ======================================================

def extract_embedding(model, sequence: str):
    """
    Extracts embeddings from the ESM-C model for a single sequence.
    """
    protein = ESMProtein(sequence)
    # The SDK handles tensor movement, but ensure model is on correct device
    protein_tensor = model.encode(protein)
    
    output = model.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    
    if not isinstance(output, LogitsOutput) or output.embeddings is None:
        raise ValueError("The model failed to return embeddings.")
        
    # Squeeze to remove batch dimension -> (Sequence_Length, Embedding_Dim)
    return output.embeddings.squeeze(0).cpu().detach().numpy()

def pad_sequence(seq: str, max_len: int, pad_char: str = "X"):
    """
    Ensures the sequence matches the fixed MAX_LENGTH.
    """
    return seq[:max_len] + pad_char * max(0, max_len - len(seq))

def run_extraction():
    print("--- FA-Amy Feature Extraction (ESM-C) ---")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading pre-trained model: {MODEL_NAME}...")
    model = ESMC.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 2. Load Dataset
    if not os.path.exists(INPUT_FASTA_PATH):
        print(f"Error: Input file not found at {INPUT_FASTA_PATH}")
        return

    records = list(SeqIO.parse(INPUT_FASTA_PATH, "fasta"))
    sequences = [str(record.seq) for record in records]
    print(f"Total sequences loaded: {len(sequences)}")

    # 3. Extraction Loop
    embedding_list = []
    print("Starting extraction (this may take a while)...")
    
    # tqdm provides a professional progress bar
    for seq in tqdm(sequences, desc="Encoding Progress"):
        padded_seq = pad_sequence(seq, MAX_LENGTH)
        try:
            with torch.no_grad():
                emb = extract_embedding(model, padded_seq)
                embedding_list.append(emb)
        except Exception as e:
            print(f"\nError processing sequence: {e}")
            continue

    # 4. Save Output
    if embedding_list:
        embedding_array = np.stack(embedding_list)
        print(f"\nExtraction complete. Final array shape: {embedding_array.shape}")
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(OUTPUT_NPY_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        np.save(OUTPUT_NPY_PATH, embedding_array)
        print(f"✅ Features successfully saved to: {OUTPUT_NPY_PATH}")
    else:
        print("❌ Failed: No embeddings were collected.")

if __name__ == "__main__":
    run_extraction()
