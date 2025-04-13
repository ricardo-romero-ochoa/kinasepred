# ------------------------------------------------------------------------------
# KinasePred: Kinase Inhibitor Bioactivity & Toxicity Prediction Platform
# © 2025 DataBiotica. All rights reserved.
# Unauthorized use, copying, or distribution is prohibited without permission.
# ------------------------------------------------------------------------------




import streamlit as st
import pandas as pd
from pipeline import main as run_pipeline  # This should accept 3 model paths
from PIL import Image

st.set_page_config(page_title="MolPred", layout="wide")
st.title("🧬 KinasePred: Bioactivity, BBB & hERG Prediction")

st.markdown("Enter **SMILES** manually or upload a CSV file with a 'SMILES' column.")

# Text input
smiles_input = st.text_area("Paste SMILES (one per line):", height=150)

# File upload
uploaded_file = st.file_uploader("Or upload CSV with SMILES column", type=["csv"])

# Let user optionally override default models
st.markdown("### 🧠 Model Files")
kinase_model = st.text_input("Kinase model (.h5)", value="models/kinase_model.h5")
herg_model   = st.text_input("hERG model (.h5)", value="models/herg_model.h5")
bbb_model    = st.text_input("BBB model (.h5)", value="models/bbb_model.h5")

# Run button
if st.button("🚀 Run Predictions"):
    with st.spinner("Running pipeline..."):
        try:
            # Build input dataframe
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            elif smiles_input.strip():
                df = pd.DataFrame({"SMILES": smiles_input.strip().splitlines()})
            else:
                st.warning("⚠️ Please provide SMILES.")
                st.stop()

            # Save temp input
            df.to_csv("input_temp.csv", index=False)

            # Run pipeline with all 3 models
            run_pipeline(
                input_smiles_csv="input_temp.csv",
                model_path=kinase_model,
                output_prefix="results",  # you can make this dynamic too
                herg_model_path=herg_model,
                bbb_model_path=bbb_model
            )

            # Load and display result
            df_result = pd.read_csv("results_final.csv")
            st.success("✅ Prediction complete!")
            st.dataframe(df_result)

            # Download
            st.download_button(
                label="📥 Download Results as CSV",
                data=df_result.to_csv(index=False),
                file_name="MolPred_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Set custom favicon and page config
st.set_page_config(
    page_title="KinasePred",
    page_icon="🧬",  # Or use a .ico path: "assets/favicon.ico"
    layout="wide"
)

# Show your logo image
logo = Image.open("assets/logo.png")
st.image(logo, width=150)

st.title("🧬 KinasePred")
st.caption("Bioactivity, BBB, and hERG prediction pipeline")

