import os
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


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
    print("📦 Loading bioactivity model...")
    model = load_model(model_path)

    smiles = df["SMILES"].tolist()
    descriptors, fps = [], []

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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if X_scaled.shape[0] == 1:
        print("⚠️ Only 1 molecule — duplicating to prevent Broken pipe.")
        X_scaled = np.vstack([X_scaled, X_scaled])

    # ✅ Check model input shape
    input_shape = model.input_shape
    print("📏 Model expects:", input_shape)
    print("📦 Input shape before reshape:", X_scaled.shape)

    if len(input_shape) == 3:
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    print("🚀 Predicting bioactivity...")
    try:
        y_pred = model.predict(X_scaled)
    except Exception as e:
        print("❌ Bioactivity prediction failed:", str(e))
        raise

    y_bin = (y_pred > 0.5).astype(int).flatten()
    df["Bioactivity"] = ["active" if x == 1 else "inactive" for x in y_bin[:len(df)]]
    df.to_csv(output_file, index=False)
    print(f"✅ Bioactivity predictions saved: {output_file}")
    return df


def apply_herg_model(df, model_path, output_file):
    print("📦 Loading hERG model...")
    X = generate_herg_descriptors(df["SMILES"].tolist())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if X_scaled.shape[0] == 1:
        print("⚠️ Only 1 molecule — duplicating for hERG prediction.")
        X_scaled = np.vstack([X_scaled, X_scaled])

    print("🚀 Predicting hERG...")
    try:
        model = load_model(model_path)
        y_pred = model.predict(X_scaled)
    except Exception as e:
        print("❌ hERG prediction failed:", str(e))
        raise

    y_bin = (y_pred > 0.5).astype(int).flatten()
    df["HERG_Toxicity"] = ["toxic" if x == 1 else "non-toxic" for x in y_bin[:len(df)]]
    df.to_csv(output_file, index=False)
    print(f"✅ hERG predictions saved: {output_file}")
    return df


def apply_bbb_model(df, model_path, output_file):
    print("📦 Loading BBB model...")
    X = generate_herg_descriptors(df["SMILES"].tolist())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if X_scaled.shape[0] == 1:
        print("⚠️ Only 1 molecule — duplicating for BBB prediction.")
        X_scaled = np.vstack([X_scaled, X_scaled])

    print("🚀 Predicting BBB permeability...")
    try:
        model = load_model(model_path)
        y_pred = model.predict(X_scaled)
    except Exception as e:
        print("❌ BBB prediction failed:", str(e))
        raise

    y_bin = (y_pred > 0.5).astype(int).flatten()
    df["BBB_permeability"] = ["permeable" if x == 1 else "non-permeable" for x in y_bin[:len(df)]]
    df.to_csv(output_file, index=False)
    print(f"✅ BBB predictions saved: {output_file}")
    return df


def main(input_smiles_csv, model_path, herg_model_path, bbb_model_path, output_prefix):
    print(f"📂 Reading input: {input_smiles_csv}")
    df = pd.read_csv(input_smiles_csv)
    print("🔢 Input shape:", df.shape)

    df = apply_keras_model(df, model_path, f"{output_prefix}_bioactivity.csv")
    df = apply_herg_model(df, herg_model_path, f"{output_prefix}_herg.csv")
    df = apply_bbb_model(df, bbb_model_path, f"{output_prefix}_final.csv")

    print("🎉 Pipeline complete.")
    return df


if __name__ == "__main__":
    main(
        input_smiles_csv="input_temp.csv",
        model_path="models/kinase_model.h5",
        herg_model_path="models/herg_model.h5",
        bbb_model_path="models/bbb_model.h5",
        output_prefix="results"
    )

