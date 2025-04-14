# ------------------------------------------------------------------------------
# KinasePred: Kinase Inhibitor Bioactivity & Toxicity Prediction Platform
# © 2025 DataBiotica. All rights reserved.
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from pipeline import main as run_pipeline
from PIL import Image
import plotly.graph_objects as go
import time

# ✅ Must be first
st.set_page_config(
    page_title="KinasePred",
    page_icon="🧬",
    layout="wide"
)

# ✅ Session state init
if "results" not in st.session_state:
    st.session_state.results = None
if "smiles_input" not in st.session_state:
    st.session_state.smiles_input = ""

# ✅ Branding
logo = Image.open("assets/logo.png")
st.image(logo, width=150)

st.title("🧬 KinasePred: Bioactivity, BBB & hERG Prediction")
st.caption("Kinase inhibition prediction pipeline including BBB permeability and hERG toxicity evaluation.")

# ----------------------------------------------------------------------
# Input Section
# ----------------------------------------------------------------------
smiles_input = st.text_area(
    "Paste SMILES (one per line):",
    height=150,
    key="smiles_input"
)

uploaded_file = st.file_uploader("📁 Upload a CSV file with a `SMILES` column", type=["csv"], key="uploaded_file")

if uploaded_file:
    st.success("✅ File uploaded successfully!")

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

        st.session_state.results = df_result

        st.download_button(
            label="📥 Download Results as CSV",
            data=df_result.to_csv(index=False),
            file_name="KinasePred_results.csv",
            mime="text/csv",
            key="download_button_main"
        )


    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        progress.empty()

# ----------------------------------------------------------------------
# Radar Plot Function
# ----------------------------------------------------------------------
def plot_radar(row):
    labels = ['MW', 'XLOGP', 'TPSA', 'HBD', 'RotatableBonds']
    values = []

    # Normalization factors (adjust as needed)
    max_vals = {
        'MW': 600,
        'XLOGP': 6,
        'TPSA': 150,
        'HBD': 5,
        'RotatableBonds': 15
    }

    for key in labels:
        val = row.get(key, 0)
        max_val = max_vals.get(key, 1)
        norm_val = min(val / max_val, 1.0) if pd.notnull(val) else 0
        values.append(norm_val)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=row['SMILES'][:12] + "..."
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="📊 Normalized Molecular Property Radar"
    )

    return fig

# ----------------------------------------------------------------------
# Visualization + Reset Button
# ----------------------------------------------------------------------

if st.session_state.results is not None:
    st.markdown("### 🧭 Radar Chart for a Molecule")

    idx = st.selectbox(
        "Select a molecule to visualize:",
        st.session_state.results.index,
        format_func=lambda i: st.session_state.results.loc[i, 'SMILES'][:30]
    )

    fig = plot_radar(st.session_state.results.loc[idx])
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="📥 Download Results as CSV",
        data=st.session_state.results.to_csv(index=False),
        file_name="KinasePred_results.csv",
        mime="text/csv",
        key="download_button_radar"
    )


    if st.button("🔄 Clear Results & Start New Batch"):
        del st.session_state.results
        st.rerun()

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 DataBiotica / KinasePred. All rights reserved.")

