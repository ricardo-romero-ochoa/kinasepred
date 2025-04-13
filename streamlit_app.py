# ------------------------------------------------------------------------------
# KinasePred: Kinase Inhibitor Bioactivity & Toxicity Prediction Platform
# © 2025 DataBiotica. All rights reserved.
# Unauthorized use, copying, or distribution is prohibited without permission.
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from pipeline import main as run_pipeline  # This should accept 3 model paths
from PIL import Image
import time

# ✅ Must be first
st.set_page_config(
    page_title="KinasePred",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = None


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
# Run Predictions
# ----------------------------------------------------------------------
if st.button("🚀 Run Predictions"):
    progress = st.progress(0, text="Initializing...")

    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        elif smiles_input.strip():
            df = pd.DataFrame({"SMILES": smiles_input.strip().splitlines()})
        else:
            st.warning("⚠️ Please provide SMILES input.")
            st.stop()

        df.to_csv("input_temp.csv", index=False)

        # Progress steps
        progress.progress(10, text="Validating molecules...")
        time.sleep(0.5)

        progress.progress(30, text="Running predictions...")
        run_pipeline(
            input_smiles_csv="input_temp.csv",
            model_path="models/kinase_model.h5",
            output_prefix="results",
            herg_model_path="models/herg_model.h5",
            bbb_model_path="models/bbb_model.h5"
        )

        progress.progress(90, text="Loading results...")
        df_result = pd.read_csv("results_final.csv")

        progress.progress(100, text="✅ Done!")
        st.success("Prediction complete!")
        st.dataframe(df_result)

        # Store results in session state for future reference
        st.session_state.results = df_result

        # Download button
        st.download_button(
            label="📥 Download Results as CSV",
            data=df_result.to_csv(index=False),
            file_name="KinasePred_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        progress.empty()

# ----------------------------------------------------------------------
# Clear Results Section
# ----------------------------------------------------------------------
if st.session_state.results is not None:
    st.markdown("---")
    st.download_button(
        label="📥 Download Results as CSV",
        data=st.session_state.results.to_csv(index=False),
        file_name="KinasePred_results.csv",
        mime="text/csv"
    )

    # Add a button to clear results and start a new batch
    if st.button("🔄 Clear Results & Start New Batch"):
        st.session_state.results = None
        st.experimental_rerun()

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 DataBiotica / KinasePred. All rights reserved.")



