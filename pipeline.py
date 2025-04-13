# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# KinasePred: Kinase Inhibitor Bioactivity & Toxicity Prediction Platform
# © 2025 DataBiotica. All rights reserved.
# ------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, Crippen, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import FilterCatalog
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- 1. VALIDATION ---
class MoleculeValidator:
    def __init__(self):
        self.results = {}

    def validate_structure(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        Chem.SanitizeMol(mol)
        return True

    def optimize_3d(self, mol):
        mol_3d = Chem.AddHs(mol)
        try:
            res = Chem.rdDistGeom.EmbedMolecule(mol_3d, randomSeed=42)
            if res == -1:
                self.results['3d_generation'] = 'Failed'
                self.results['3d_optimization'] = 'Failed'
                return None
            self.results['3d_generation'] = 'Success'
            res = Chem.rdForceFieldHelpers.MMFFOptimizeMolecule(mol_3d)
            self.results['3d_optimization'] = 'Success' if res != -1 else 'Failed'
            return mol_3d
        except:
            self.results['3d_generation'] = 'Failed'
            self.results['3d_optimization'] = 'Failed'
            return None

    def check_stereochemistry(self, mol):
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return all(c[1] != '?' for c in chiral_centers)

    def analyze_tautomers(self, mol):
        enumerator = rdMolStandardize.TautomerEnumerator()
        return len(enumerator.Enumerate(mol))

    def validate_molecule(self, smiles):
        self.results = {}
        if not self.validate_structure(smiles):
            self.results['valid_structure'] = False
            return self.results
        self.results['valid_structure'] = True
        mol = Chem.MolFromSmiles(smiles)
        mol_3d = self.optimize_3d(mol)
        self.results['stereochemistry_defined'] = self.check_stereochemistry(mol) if mol_3d else 'N/A'
        self.results['tautomer_count'] = self.analyze_tautomers(mol)
        return self.results

def validate_csv(input_file, output_file):
    validator = MoleculeValidator()
    df_input = pd.read_csv(input_file)
    if 'SMILES' not in df_input.columns:
        raise ValueError("Missing 'SMILES' column.")
    results = []
    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Validating"):
        res = validator.validate_molecule(row['SMILES'])
        results.append({**row.to_dict(), **res})
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    return df_out

# --- 2. LIPINSKI FILTERS ---
def calculate_rule_violations(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: 1 for k in ['Lipinski', 'Ghose', 'Veber', 'Egan', 'Muegge']}
    return {
        'Lipinski': int(Lipinski.NumHAcceptors(mol) > 10 or Lipinski.NumHDonors(mol) > 5 or Descriptors.MolLogP(mol) > 5 or Descriptors.MolWt(mol) > 500),
        'Ghose': int(Crippen.MolMR(mol) > 130 or Crippen.MolMR(mol) < 40 or Descriptors.MolWt(mol) > 480 or Descriptors.MolWt(mol) < 160),
        'Veber': int(Descriptors.NumRotatableBonds(mol) > 10 or Descriptors.TPSA(mol) > 140),
        'Egan': int(Descriptors.MolLogP(mol) > 5.88 or Descriptors.TPSA(mol) > 131.6),
        'Muegge': int(Descriptors.MolWt(mol) < 200 or Descriptors.MolWt(mol) > 600 or Lipinski.NumHDonors(mol) > 5 or Lipinski.NumHAcceptors(mol) > 10)
    }

def apply_lipinski_rules(df, output_file):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Lipinski Filters"):
        res = calculate_rule_violations(row['SMILES'])
        results.append({**row.to_dict(), **res})
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    return df_out

# --- 3. LEAD-LIKENESS ---
def check_leadlikeness(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return (False, None, None, None)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    return (250 <= mw <= 350 and logp <= 3.5 and rotb <= 7, mw, logp, rotb)

def apply_leadlikeness(df, output_file):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Lead-likeness"):
        ll, mw, logp, rotb = check_leadlikeness(row['SMILES'])
        results.append({**row.to_dict(), "Leadlike": ll, "MW": mw, "XLOGP": logp, "RotatableBonds": rotb})
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    return df_out

# --- 4. BRENK & PAINS ---
def analyze_brenk_pains(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return {'brenk_alerts': None, 'pains_alerts': None}
    brenk = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)).HasMatch(mol)
    pains = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)).HasMatch(mol)
    return {'brenk_alerts': brenk, 'pains_alerts': pains}

def apply_brenk_pains(df, output_file):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Brenk/PAINS"):
        res = analyze_brenk_pains(row['SMILES'])
        results.append({**row.to_dict(), **res})
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    return df_out

# --- 5. ML MODEL WRAPPERS ---

import os
import numpy as np
import pandas as pd
import logging
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ✅ Setup logging
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def generate_herg_descriptors(smiles_list):
    data_rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            data_rows.append([0] * (6 + 2048 + 167))
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        n_aromatic = Lipinski.NumAromaticRings(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        numeric = [mw, logp, tpsa, hbd, n_aromatic, rot_bonds]

        rdk_fp = Chem.RDKFingerprint(mol, fpSize=2048)
        rdk_bits = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(rdk_fp, rdk_bits)

        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_bits = np.zeros((167,), dtype=int)
        DataStructs.ConvertToNumpyArray(maccs_fp, maccs_bits)

        row = numeric + rdk_bits.tolist() + maccs_bits.tolist()
        data_rows.append(row)

    return np.array(data_rows, dtype=float)

def apply_keras_model(df, model_path, output_file):
    logging.info("Loading bioactivity model...")
    model = load_model(model_path)

    smiles = df["SMILES"].tolist()
    descriptors = []
    fps = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            descriptors.append([0, 0, 0, 0])
            fps.append([0] * 2048)
            continue
        descriptors.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol)
        ])
        fp = Chem.RDKFingerprint(mol, fpSize=2048)
        arr = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    X = np.hstack([descriptors, fps])
    if X.shape[0] == 1:
        logging.warning("Only one molecule detected — duplicating row.")
        X = np.vstack([X, X])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    try:
        y_pred = model.predict(X_scaled)
    except Exception as e:
        logging.error(f"Bioactivity prediction failed: {e}")
        raise

    y_bin = (y_pred > 0.5).astype(int).flatten()
    df["Bioactivity"] = ["active" if x == 1 else "inactive" for x in y_bin[:len(df)]]
    df.to_csv(output_file, index=False)
    logging.info(f"Saved bioactivity predictions: {output_file}")
    return df

def apply_herg_model(df, model_path, output_file):
    logging.info("Loading hERG model...")
    X = generate_herg_descriptors(df["SMILES"].tolist())
    if X.shape[0] == 1:
        logging.warning("Only one molecule detected — duplicating row.")
        X = np.vstack([X, X])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = load_model(model_path)
        y_pred = model.predict(X_scaled)
    except Exception as e:
        logging.error(f"hERG prediction failed: {e}")
        raise

    y_bin = (y_pred > 0.5).astype(int).flatten()
    df["HERG_Toxicity"] = ["toxic" if x == 1 else "non-toxic" for x in y_bin[:len(df)]]
    df.to_csv(output_file, index=False)
    logging.info(f"Saved hERG predictions: {output_file}")
    return df

def apply_bbb_model(df, model_path, output_file):
    logging.info("Loading BBB model...")
    X = generate_herg_descriptors(df["SMILES"].tolist())
    if X.shape[0] == 1:
        logging.warning("Only one molecule detected — duplicating row.")
        X = np.vstack([X, X])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = load_model(model_path)
        y_pred = model.predict(X_scaled)
    except Exception as e:
        logging.error(f"BBB prediction failed: {e}")
        raise

    y_bin = (y_pred > 0.5).astype(int).flatten()
    df["BBB_permeability"] = ["permeable" if x == 1 else "non-permeable" for x in y_bin[:len(df)]]
    df.to_csv(output_file, index=False)
    logging.info(f"Saved BBB predictions: {output_file}")
    return df

def clean_intermediate_files(prefix):
    keep = f"{prefix}_final.csv"
    for fname in os.listdir():
        if fname.startswith(prefix) and fname != keep and fname.endswith(".csv"):
            os.remove(fname)
            logging.info(f"🧹 Deleted intermediate file: {fname}")

def main(input_smiles_csv, model_path, herg_model_path, bbb_model_path, output_prefix):
    logging.info(f"🚀 Running pipeline on: {input_smiles_csv}")
    df = pd.read_csv(input_smiles_csv)
    logging.info(f"Input shape: {df.shape}")

    df = apply_keras_model(df, model_path, f"{output_prefix}_bioactivity.csv")
    df = apply_herg_model(df, herg_model_path, f"{output_prefix}_herg.csv")
    df = apply_bbb_model(df, bbb_model_path, f"{output_prefix}_final.csv")

    clean_intermediate_files(output_prefix)

    logging.info("✅ Pipeline finished.")
    return df

if __name__ == "__main__":
    main(
        input_smiles_csv="input_temp.csv",
        model_path="models/kinase_model.h5",
        herg_model_path="models/herg_model.h5",
        bbb_model_path="models/bbb_model.h5",
        output_prefix="results"
    )
