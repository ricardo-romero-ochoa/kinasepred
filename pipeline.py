# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# KinasePred: Kinase Inhibitor Bioactivity & Toxicity Prediction Platform
# © 2025 DataBiotica. All rights reserved.
# Unauthorized use, copying, or distribution is prohibited without permission.
# ------------------------------------------------------------------------------


"""Validation_pipeline.ipynb

Author: Dr. Ricardo Romero
Natural Sciences Depatrment
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
import rdkit # This line is added to import the rdkit module
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, rdChemReactions
from rdkit.Chem import Lipinski, Crippen, RDConfig
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import FilterCatalog
from rdkit.Chem import MACCSkeys

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now sascore can be imported
import sascorer

# TensorFlow / Keras imports
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


# -----------------------------------------
# 1) Molecule Validation (Validation_script)
# -----------------------------------------
class MoleculeValidator:
    """
    Performs basic checks (SMILES validity, stereochemistry),
    3D structure embedding + forcefield optimization,
    tautomer counting, and synthetic accessibility scoring.
    """

    def __init__(self):
        self.results = {}

    def validate_structure(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)  # raises exception if invalid
        return True

    def optimize_3d(self, mol):
        """
        Generates a 3D conformation and attempts an MMFF optimization.
        """
        mol_3d = Chem.AddHs(mol)
        try:
            # Attempt embedding
            embed_result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            if embed_result == -1:
                self.results['3d_generation'] = 'Failed: Could not embed molecule'
                # If embedding fails, no optimization is attempted
                self.results['3d_optimization'] = 'Failed'
                return None
            else:
                # Embedding succeeded
                self.results['3d_generation'] = 'Success'

            # Attempt optimization
            opt_result = AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
            if opt_result == -1:
                self.results['3d_optimization'] = 'Failed: Optimization did not converge'
            else:
                self.results['3d_optimization'] = 'Success'

            return mol_3d

        except Exception as e:
            self.results['3d_generation'] = f'Failed: {str(e)}'
            self.results['3d_optimization'] = 'Failed'
            return None

    def check_stereochemistry(self, mol):
        """
        Checks for undefined stereochemistry.
        Considers both '?' and centers with implicit hydrogens as undefined.
        Returns True if well-defined, False otherwise.
        """
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        undefined_centers = [
            center
            for center, chirality in chiral_centers
            if chirality == '?' or chirality == 'u'  # Include undefined or unknown chirality
        ]
        if undefined_centers:  # If there are undefined chiral centers
            return False  # Not well-defined stereochemistry
        else:
            return True  # Well-defined stereochemistry

    def analyze_tautomers(self, mol):
        """
        Returns how many tautomeric forms are found by rdMolStandardize.
        """
        enumerator = rdMolStandardize.TautomerEnumerator()
        tautomers = enumerator.Enumerate(mol)
        return len(tautomers)

    def calculate_synthetic_accessibility(self, mol):
        """
        Returns the Ertl–Schuffenhauer synthetic accessibility score
        from ~1 (easiest) to ~10 (hardest).
        """
        return sascorer.calculateScore(mol)

    def validate_molecule(self, smiles):
        """
        Returns a dictionary of results for each molecule:
         - valid_structure
         - 3d_generation
         - 3d_optimization
         - stereochemistry_defined
         - tautomer_count
         - sa_score
        """
        self.results = {}

        # Step 1: SMILES / structure check
        if not self.validate_structure(smiles):
            self.results['valid_structure'] = False
            # Don't attempt further steps
            return self.results
        self.results['valid_structure'] = True

        mol = Chem.MolFromSmiles(smiles)

        # Step 2: 3D structure + optimization
        mol_3d = self.optimize_3d(mol)
        if mol_3d is not None:
            self.results['stereochemistry_defined'] = self.check_stereochemistry(mol_3d)
        else:
            # If 3D gen/opt fails, we can’t reliably check stereochemistry
            self.results['stereochemistry_defined'] = 'N/A'

        # Step 4: Tautomer analysis
        self.results['tautomer_count'] = self.analyze_tautomers(mol)

        # Step 5: Synthetic accessibility (optional, if you want to re-enable)
        self.results['sa_score'] = self.calculate_synthetic_accessibility(mol)

        return self.results


def validate_csv(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Reads SMILES from `input_file`, validates each with MoleculeValidator,
    saves results as `output_file`, and returns a DataFrame of results.
    """
    validator = MoleculeValidator()
    df_input = pd.read_csv(input_file)
    if 'SMILES' not in df_input.columns:
        raise ValueError("Input CSV must have a 'SMILES' column")

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


# -----------------------------------------
# 2) Lipinski & other filters (Lipinski.pdf)
# -----------------------------------------
def calculate_rule_violations(smiles):
    """
    For a single SMILES:
      - Lipinski
      - Ghose
      - Veber
      - Egan
      - Muegge
    Returns a dict with each rule flagged 0/1 (0 = pass, 1 = violation).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # If invalid, mark them all as 'bad' or just 1
        return {'Lipinski': 1, 'Ghose': 1, 'Veber': 1, 'Egan': 1, 'Muegge': 1}

    # -- Lipinski's rule
    lipinski_fail = (
        Lipinski.NumHAcceptors(mol) > 10 or
        Lipinski.NumHDonors(mol) > 5 or
        Descriptors.MolLogP(mol) > 5 or
        Descriptors.MolWt(mol) > 500
    )

    # -- Ghose filter
    ghose_fail = (
        Crippen.MolMR(mol) > 130 or Crippen.MolMR(mol) < 40 or
        Descriptors.MolLogP(mol) > 5.6 or Descriptors.MolLogP(mol) < -0.4 or
        Descriptors.MolWt(mol) > 480 or Descriptors.MolWt(mol) < 160 or
        mol.GetNumAtoms() > 70 or mol.GetNumAtoms() < 20
    )

    # -- Veber's rule
    veber_fail = (
        Descriptors.NumRotatableBonds(mol) > 10 or
        rdMolDescriptors.CalcTPSA(mol) > 140
    )

    # -- Egan's rule
    egan_fail = (
        Descriptors.MolLogP(mol) > 5.88 or
        rdMolDescriptors.CalcTPSA(mol) > 131.6
    )

    # -- Muegge's rule
    muegge_fail = (
        Descriptors.MolLogP(mol) < -0.4 or Descriptors.MolLogP(mol) > 5.6 or
        Descriptors.MolWt(mol) < 200 or Descriptors.MolWt(mol) > 600 or
        Lipinski.NumHDonors(mol) > 5 or Lipinski.NumHAcceptors(mol) > 10
    )

    return {
        'Lipinski': int(lipinski_fail),
        'Ghose': int(ghose_fail),
        'Veber': int(veber_fail),
        'Egan': int(egan_fail),
        'Muegge': int(muegge_fail)
    }


def apply_lipinski_rules(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Takes a DataFrame with SMILES, calculates rule violations,
    merges them, writes to `output_file`, returns merged DataFrame.
    """
    if 'SMILES' not in df.columns:
        raise ValueError("DataFrame must have SMILES column")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Lipinski+ filters"):
        smiles = row['SMILES']
        r = calculate_rule_violations(smiles)
        rec = {**row.to_dict(), **r}
        records.append(rec)

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_file, index=False)
    print(f"[2] Lipinski+ filters complete. Saved: {output_file}")
    return out_df


# -----------------------------------------
# 3) Lead-likeness (Lead-likeness.pdf)
# -----------------------------------------
def check_leadlikeness(smiles):
    """
    Check typical leadlike ranges:
    - 250 <= MW <= 350
    - LogP <= 3.5
    - <= 7 rotatable bonds
    Returns (bool, MW, xlogp, rotb).
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return (False, None, None, None)

    mw = Descriptors.MolWt(mol)
    xlogp = Descriptors.MolLogP(mol)
    rotb = Descriptors.NumRotatableBonds(mol)

    is_leadlike = (
        (250 <= mw <= 350) and
        (xlogp <= 3.5) and
        (rotb <= 7)
    )
    return (is_leadlike, mw, xlogp, rotb)


def apply_leadlikeness(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Reads from df, calculates lead-likeness, merges, writes to output_file.
    """
    if 'SMILES' not in df.columns:
        raise ValueError("DataFrame must have SMILES column")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Lead-likeness"):
        smiles = row['SMILES']
        leadlike, mw, xlogp, rotb = check_leadlikeness(smiles)
        new_fields = {
            'Leadlike': leadlike,
            'MW': mw,
            'XLOGP': xlogp,
            'RotatableBonds': rotb
        }
        combined = {**row.to_dict(), **new_fields}
        records.append(combined)

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_file, index=False)
    print(f"[3] Lead-likeness analysis complete. Saved: {output_file}")
    return out_df

# -----------------------------------------
# 4) Brenk & PAINS (Brenk.pdf)
# -----------------------------------------

def analyze_brenk_pains(smiles: str):
    """
    Applies Brenk and PAINS checks to a single SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {'brenk_alerts': None, 'pains_alerts': None}

    # Brenk
    brenk_catalog_params = FilterCatalog.FilterCatalogParams(
        FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK
    )
    brenk_catalog = FilterCatalog.FilterCatalog(brenk_catalog_params)
    brenk_alert = brenk_catalog.HasMatch(mol)

    # PAINS
    pains_catalog_params = FilterCatalog.FilterCatalogParams(
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS
    )
    pains_catalog = FilterCatalog.FilterCatalog(pains_catalog_params)
    pains_alert = pains_catalog.HasMatch(mol)

    return {
        'brenk_alerts': brenk_alert,
        'pains_alerts': pains_alert
    }

def apply_brenk_pains(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Applies Brenk and PAINS analysis to a DataFrame of SMILES.

    Args:
        df: DataFrame containing a 'SMILES' column.
        output_file: Path to save the final CSV file.

    Returns:
        DataFrame with added columns for:
          - brenk_alerts
          - pains_alerts
    """
    if 'SMILES' not in df.columns:
        raise ValueError("DataFrame must have a 'SMILES' column")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Brenk/PAINS"):
        smiles = row['SMILES']
        results = analyze_brenk_pains(smiles)  # Brenk + PAINS only
        combined = {**row.to_dict(), **results}
        records.append(combined)

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_file, index=False)
    print(f"[4] Brenk/PAINS analysis complete. Saved: {output_file}")

    return df_out

# -----------------------------------------
# 5) Apply Model (Apply model.pdf)
# -----------------------------------------
def generate_lipinski_descriptors(smiles_list):
    """
    For each SMILES, generate [MW, LogP, NumHDonors, NumHAcceptors].
    Return a DataFrame with these columns for each SMILES.
    """
    data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            data.append([None, None, None, None])
            continue
        data.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol)
        ])
    df_desc = pd.DataFrame(data, columns=["MW_calc", "LogP_calc", "NumHDonors_calc", "NumHAcceptors_calc"])
    return df_desc


def apply_keras_model(df: pd.DataFrame, model_path: str, output_file: str) -> pd.DataFrame:
    """
    Loads a Keras model, generates descriptors+fingerprints,
    predicts binary activity, merges results, writes final CSV.
    """
    # 1) Load model
    model = load_model(model_path)
    print("[5] Keras model loaded.")

    # 2) Generate or gather descriptors
    smi_list = df['SMILES'].tolist()
    lip_desc_df = generate_lipinski_descriptors(smi_list)

    # Combine them with RDKFingerprint
    rdk_fingerprints = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            rdk_fingerprints.append([0]*2048)
            continue
        fp = Chem.RDKFingerprint(mol, fpSize=2048)
        arr = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rdk_fingerprints.append(arr)

    rdk_fingerprints = np.array(rdk_fingerprints, dtype=float)
    combined_data = np.hstack([lip_desc_df.values, rdk_fingerprints])

    # 3) Scale the combined data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)

    # 4) Reshape if needed (depending on your model’s input shape)
    combined_data_scaled = combined_data_scaled.reshape(
        (combined_data_scaled.shape[0], combined_data_scaled.shape[1], 1)
    )

    # 5) Predict
    y_pred = model.predict(combined_data_scaled)
    y_bin = (y_pred > 0.5).astype(int).flatten()

    # 6) Build a "Class" column => "active" or "non-active"
    class_labels = ["active" if x == 1 else "inactive" for x in y_bin]

    # 7) Append to DataFrame
    #df['Binary_activity'] = y_bin
    df['Bioactivity'] = class_labels

    df.to_csv(output_file, index=False)
    print(f"[5] Model predictions complete. Bioactivity results saved: {output_file}")
    return df



def generate_herg_descriptors(smiles_list):
    """
    For each SMILES, compute:
      - MW
      - LogP
      - TPSA
      - HBD (NumHDonors)
      - n_aromatic_rings
      - RotatableBonds
      - RDKFingerprint (2048 bits)
      - MACCSkeys (167 bits)

    Return a numpy array of shape (N, 6 + 2048 + 167).
    """
    data_rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            # If molecule is invalid, append zeros for descriptors and fingerprints
            data_rows.append([0] * (6 + 2048 + 167))  # 6 descriptors + 2048 RDKFingerprint bits + 167 MACCSkeys bits
            continue

        # 1) Numeric descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)  # or rdMolDescriptors.CalcTPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        n_aromatic = Lipinski.NumAromaticRings(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)

        numeric_part = [mw, logp, tpsa, hbd, n_aromatic, rot_bonds]

        # 2) RDKFingerprint
        rdk_fp = Chem.RDKFingerprint(mol, fpSize=2048)
        rdk_bits = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(rdk_fp, rdk_bits)

        # 3) MACCSkeys
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_bits = np.zeros((167,), dtype=int) # Standard size for MACCSkeys is 167
        DataStructs.ConvertToNumpyArray(maccs_fp, maccs_bits)

        # Combine all features
        row = numeric_part + rdk_bits.tolist() + maccs_bits.tolist()
        data_rows.append(row)

    return np.array(data_rows, dtype=float)




def apply_herg_model(df, herg_model_path, output_file):
    """
    Uses the hERG model (dense layers) that expects [6 numeric descriptors + 167-bit MACCS].
    Adds a new 'HERG_Toxicity' column: 'toxic' / 'non-toxic'.
    Saves the final DataFrame to 'output_file'.
    """
    # 1) Generate hERG descriptors
    smiles_list = df["SMILES"].tolist()
    X_herg = generate_herg_descriptors(smiles_list)  # shape: (N, 6+167)

    # 2) Scale them
    scaler = StandardScaler()
    X_herg_scaled = scaler.fit_transform(X_herg)  # shape: (N, 6+167)

    # 3) Load hERG model
    herg_model = load_model(herg_model_path)
    print("hERG model loaded. Input shape expected:", herg_model.input_shape)

    # 4) Predict
    # Because it's a Dense-only model with input_dim = (N, 173),
    # we do NOT reshape to (N, 173, 1). Just keep (N, 173).
    y_pred = herg_model.predict(X_herg_scaled)        # shape: (N,1)
    y_bin = (y_pred > 0.5).astype(int).flatten()       # shape: (N,)

    # Map 0 => "non-toxic", 1 => "toxic"
    labels = ["toxic" if x == 1 else "non-toxic" for x in y_bin]

    # 5) Append to DataFrame
    df["HERG_Toxicity"] = labels

    # 6) Save final
    df.to_csv(output_file, index=False)
    print(f"hERG predictions complete. Results saved to {output_file}")

    return df

def apply_bbb_model(df, bbb_model_path, output_file):
    """
    Uses the hERG model (dense layers) that expects [6 numeric descriptors + 167-bit MACCS].
    Adds a new 'HERG_Toxicity' column: 'toxic' / 'non-toxic'.
    Saves the final DataFrame to 'output_file'.
    """
    # 1) Generate hERG descriptors
    smiles_list = df["SMILES"].tolist()
    X_herg = generate_herg_descriptors(smiles_list)  # shape: (N, 6+167)

    # 2) Scale them
    scaler = StandardScaler()
    X_herg_scaled = scaler.fit_transform(X_herg)  # shape: (N, 6+167)

    # 3) Load BBB model
    bbb_model = load_model(bbb_model_path)
    print("BBB model loaded. Input shape expected:", bbb_model.input_shape)

    # 4) Predict
    # Because it's a Dense-only model with input_dim = (N, 173),
    # we do NOT reshape to (N, 173, 1). Just keep (N, 173).
    y_pred = bbb_model.predict(X_herg_scaled)        # shape: (N,1)
    y_bin = (y_pred > 0.5).astype(int).flatten()       # shape: (N,)

    # Map 0 => "non-permeable", 1 => "permeable"
    labels = ["permeable" if x == 1 else "non-permeable" for x in y_bin]

    # 5) Append to DataFrame
    df["BBB_permeability"] = labels

    # 6) Save final
    df.to_csv(output_file, index=False)
    print(f"BBB predictions complete. Results saved to {output_file}")

    return df



# -----------------------------------------
# MAIN / Pipeline
# -----------------------------------------
def main(
    input_smiles_csv=str,
    model_path=str,
    herg_model_path="models/herg_model.h5",
    bbb_model_path="model/bbb_model.h5",
    output_prefix=str
):
    """
    Runs the entire pipeline on the given CSV of SMILES:
      1) Validation
      2) Lipinski & other filters
      3) Lead-likeness
      4) Brenk / PAINS / BBB
      5) Model inference (Keras)
    """

    # 1) Validate
    val_csv = f"{output_prefix}_validation.csv"
    df_step1 = validate_csv(input_smiles_csv, val_csv)

    # 2) Lipinski + filters
    lip_csv = f"{output_prefix}_lipinski.csv"
    df_step2 = apply_lipinski_rules(df_step1, lip_csv)

    # 3) Lead-likeness
    lead_csv = f"{output_prefix}_lead.csv"
    df_step3 = apply_leadlikeness(df_step2, lead_csv)

    # 4) Brenk/PAINS/BBB
    brenk_csv = f"{output_prefix}_brenk.csv"
    df_step4 = apply_brenk_pains(df_step3, brenk_csv)

    # 5) Bioactivity Model inference
    bioactivity_csv = f"{output_prefix}_bioactivity.csv"
    df_step5 = apply_keras_model(df_step4, model_path, bioactivity_csv)

    # 6) Herg Model inference
    herg_csv = f"{output_prefix}_herg.csv"
    df_step6 = apply_herg_model(df_step5, herg_model_path, herg_csv)

    # 7) BBB Model inference
    final_csv = f"{output_prefix}_final.csv"
    df_final = apply_bbb_model(df_step6, bbb_model_path, final_csv)

    print("\n=== Pipeline Complete! ===")
    print(f"Final file: {final_csv}")

# -------------------------------------------------------------------
# No sys.argv logic below, so it won't parse the command line at all.
# If you run this file directly from the terminal, it will just
# use the defaults in `main()`.
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Simply call main() with default arguments
    main()
