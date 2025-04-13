# ------------------------------------------------------------------------------
# KinasePred: Kinase Inhibitor Bioactivity & Toxicity Prediction Platform
# © 2025 DataBiotica. All rights reserved.
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from pipeline import main as run_pipeline  # Make sure pipeline.py is in same folder
from PIL import Image
import plotly.graph_objects as go
import time
import os

# ✅ Must be first
st.set_page_config(
    page_title="KinasePred",
    page_icon="🧬",
    layout="wide"
)

# ✅ App branding
logo = Image.open("assets/logo.png")
st.image(logo, width=150)
st.title("🧬 KinasePred: Bioactivity, BBB & hERG Prediction")
st.caption("Predict kinase bioactivity, hERG toxicity, and BBB permeability from SMILES.")

# --- INPUT SECTION ---
smiles_input = st.text_area("Paste SMILES (one per line):", height=150)
uploaded_file = st.file_uploader("📁 Upload a CSV file with a `SMILES` column", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

# --- RUN BUTTON ---
if st.button("🚀 Run Predictions"):
    try:
        # Step 1: Prepare input
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        elif smiles_input.strip():
            df = pd.DataFrame({"SMILES": smiles_input.strip().splitlines()})
        else:
            st.warning("⚠️ Please provide SMILES.")
            st.stop()

        # Handle single-SMILES edge case: duplicate if needed
        if df.shape[0] == 1:
            st.warning("⚠️ Only one SMILES provided — duplicating for stability.")
            df = pd.concat([df, df])

        st.write("Input shape:", df.shape)
        st.write(df.head())

        # Step 2: Save temp input
        df.to_csv("input_temp.csv", index=False)
        st.success("✅ input_temp.csv saved.")

        # Step 3: Run pipeline
        with st.spinner("Running pipeline..."):
            run_pipeline(
                input_smiles_csv="input_temp.csv",
                model_path="models/kinase_model.h5",
                herg_model_path="models/herg_model.h5",
                bbb_model_path="models/bbb_model.h5",
                output_prefix="results"
            )

        # Step 4: Load and display results
        if os.path.exists("results_final.csv"):
            df_result = pd.read_csv("results_final.csv")
            st.success("✅ Prediction complete!")
            st.dataframe(df_result)

            # Download button
            st.download_button(
                label="📥 Download Results",
                data=df_result.to_csv(index=False),
                file_name="KinasePred_results.csv",
                mime="text/csv"
            )

            # --- Radar Chart ---
            def plot_radar(row):
                labels = ['MW', 'XLOGP', 'TPSA', 'HBD', 'RotatableBonds']
                raw_values = [row.get(k, 0) for k in labels]
                max_vals = {'MW': 600, 'XLOGP': 6, 'TPSA': 150, 'HBD': 5, 'RotatableBonds': 15}
                norm = [v / max_vals[l] if max_vals[l] else 0 for v, l in zip(raw_values, labels)]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=norm, theta=labels, fill='toself', name=row['SMILES'][:12]))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
                return fig

            st.markdown("### 🧭 Radar Chart")
            idx = st.selectbox("Select a molecule:", df_result.index, format_func=lambda i: df_result.loc[i, 'SMILES'][:30])
            st.plotly_chart(plot_radar(df_result.loc[idx]), use_container_width=True)
        else:
            st.error("❌ Final result file not found.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2025 DataBiotica / KinasePred. All rights reserved.")
