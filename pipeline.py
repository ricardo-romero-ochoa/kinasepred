# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# KinasePred: Kinase Inhibitor Bioactivity & Toxicity Prediction Platform
# © 2025 DataBiotica. All rights reserved.
# Unauthorized use, copying, or distribution is prohibited without permission.
# ------------------------------------------------------------------------------

"""Validation_pipeline.ipynb

Author: Dr. Ricardo Romero
Natural Sciences Department
UAM-C
rromero@cua.uam.mx
"""

#!/usr/bin/env python3

import os
import sys
import subprocess
import math
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

# RDKit imports
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, rdChemReactions
from rdkit.Chem import Lipinski, Crippen, RDConfig
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import FilterCatalog
from rdkit.Chem import MACCSkeys

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# TensorFlow / Keras imports
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


# -----------------------------------------
# 1) Molecule Validation (Validation_script)
# -----------------------------------------
class MoleculeValidator:
    """Performs structural validation, 3D optimization, and calculates properties."""
    # ... [MoleculeValidator class remains unchanged] ...


def validate_dataframe(df_input: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """Validates molecules in the input DataFrame and saves results."""
    validator = MoleculeValidator()
    if 'SMILES' not in df_input.columns:
        raise ValueError("Input DataFrame must have a 'SMILES' column")
    
    results_list = []
    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Validating"):
        smiles = row['SMILES']
        vres = validator.validate_molecule(smiles)
        combined = {**row.to_dict(), **vres}
        results_list.append(combined)
    
    df_out = pd.DataFrame(results_list)
    df_out.to_csv(output_file, index=False)
    print(f"[1] Validation complete. Saved: {output_file}")
    return df_out


def validate_csv(input_file: str, output_file: str) -> pd.DataFrame:
    """Reads CSV and validates using validate_dataframe."""
    df_input = pd.read_csv(input_file)
    return validate_dataframe(df_input, output_file)


# ... [Rest of the functions (Lipinski, Lead-likeness, Brenk/PAINS, Model application) remain unchanged] ...


# -----------------------------------------
# MAIN / Pipeline
# -----------------------------------------
def main(
    input_smiles: str,  # Changed from input_smiles_csv to input_smiles
    model_path: str,
    herg_model_path: str = "models/herg_model.h5",
    bbb_model_path: str = "model/bbb_model.h5",
    output_prefix: str = "output"
):
    """Runs the pipeline on either a CSV of SMILES or a single SMILES string."""
    # Input handling remains the same
    if os.path.isfile(input_smiles):  # Now matches parameter name
        df_input = pd.read_csv(input_smiles)
    else:
        df_input = pd.DataFrame({'SMILES': [input_smiles]})



    # Pipeline steps
    val_csv = f"{output_prefix}_validation.csv"
    df_step1 = validate_dataframe(df_input, val_csv)

    lip_csv = f"{output_prefix}_lipinski.csv"
    df_step2 = apply_lipinski_rules(df_step1, lip_csv)

    lead_csv = f"{output_prefix}_lead.csv"
    df_step3 = apply_leadlikeness(df_step2, lead_csv)

    brenk_csv = f"{output_prefix}_brenk.csv"
    df_step4 = apply_brenk_pains(df_step3, brenk_csv)

    bioactivity_csv = f"{output_prefix}_bioactivity.csv"
    df_step5 = apply_keras_model(df_step4, model_path, bioactivity_csv)

    herg_csv = f"{output_prefix}_herg.csv"
    df_step6 = apply_herg_model(df_step5, herg_model_path, herg_csv)

    final_csv = f"{output_prefix}_final.csv"
    df_final = apply_bbb_model(df_step6, bbb_model_path, final_csv)

    print("\n=== Pipeline Complete! ===")
    print(f"Final file: {final_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run KinasePred prediction pipeline.')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path or a single SMILES string.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to bioactivity Keras model.')
    parser.add_argument('--herg_model', type=str, default='models/herg_model.h5',
                        help='Path to hERG toxicity model (default: models/herg_model.h5).')
    parser.add_argument('--bbb_model', type=str, default='model/bbb_model.h5',
                        help='Path to BBB permeability model (default: model/bbb_model.h5).')
    parser.add_argument('--output_prefix', type=str, default='output',
                        help='Prefix for output files (default: output).')
    args = parser.parse_args()

    main(
        input_smiles=args.input,
        model_path=args.model_path,
        herg_model_path=args.herg_model,
        bbb_model_path=args.bbb_model,
        output_prefix=args.output_prefix
    )
