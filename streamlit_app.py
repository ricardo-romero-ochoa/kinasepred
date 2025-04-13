# ------------------------------------------------------------------------------
# KinasePred: Kinase Inhibitor Bioactivity & Toxicity Prediction Platform
# © 2025 DataBiotica. All rights reserved.
# Unauthorized use, copying, or distribution is prohibited without permission.
# ------------------------------------------------------------------------------

import streamlit as st

# ✅ Must be first
st.set_page_config(
    page_title="KinasePred",
    page_icon="🧬",
    layout="wide"
)

import pandas as pd
from pipeline import main as run_pipeline  # This should accept 3 model paths
from PIL import Image

# ✅ Branding (after config)
logo = Image.open("assets/logo.png")
st.image(logo, width=150)

st.title("🧬 KinasePred: Bioactivity, BBB & hERG Prediction")
st.caption("Kinase inhibition prediction pipeline including BBB permeability and hERG toxicity evaluation.")

# ----------------------------------------------------------------------
# Input Section
# ----------------------------------------------------------------------
st.markdown("### 🧪 Enter **SMILES** manually or upload a CSV file:")

smiles_input = st.text_area("Paste SMILES (one per line):", height=150)

uploaded_file = st.file_uploader("Or upload CSV with a `SMILES` column", type=["csv"])

# ----------------------------------------------------------------------
# Model selection (optional paths)
# ----------------------------------------------------------------------
st.markdown("### 🧠 Model Files")
kinase_model = st.text_input("Kinase model (.h5)", value="models/kinase_model.h5")
herg_model   = st.text_input("hERG model (.h5)", value="models/herg_model.h5")
bbb_model    = st.text_input("BBB model (.h5)", value="models/bbb_model.h5")

# ----------------------------------------------------------------------
# Run button
# ----------------------------------------------------------------------
if st.button("🚀 Run Predictions"):
    with st.spinner("Running pipeline..."):
        try:
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            elif smiles_input.strip():
                df = pd.DataFrame({"SMILES": smiles_input.strip().splitlines()})
            else:
                st.warning("⚠️ Please provide SMILES input.")
                st.stop()

            df.to_csv("input_temp.csv", index=False)

            # Run main pipeline
            run_pipeline(
                input_smiles_csv="input_temp.csv",
                model_path=kinase_model,
                output_prefix="results",
                herg_model_path=herg_model,
                bbb_model_path=bbb_model
            )

            df_result = pd.read_csv("results_final.csv")
            st.success("✅ Prediction complete!")
            st.dataframe(df_result)

            # Download option
            st.download_button(
                label="📥 Download Results as CSV",
                data=df_result.to_csv(index=False),
                file_name="KinasePred_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 DataBiotica / KinasePred. All rights reserved.")


