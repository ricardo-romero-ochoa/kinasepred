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


# ✅ Initialize session state
for key in ["results", "smiles_input", "uploaded_file"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "results" else ""



# ✅ Branding (after config)
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

# Styled file drop zone
st.markdown("""
    <style>
    .file-drop-zone {
        border: 3px dashed #00c0f2;
        border-radius: 10px;
        padding: 16px;
        background-color: #f0fbff;
        animation: pulse 2s infinite;
        text-align: center;
        font-weight: 500;
        margin-top: 10px;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0,192,242, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(0,192,242, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0,192,242, 0); }
    }
    </style>
    <div class="file-drop-zone">📁 Drop or browse a CSV file here with SMILES</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv"], key="uploaded_file")


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
# Visualization
# ----------------------------------------------------------------------
def plot_radar(row):
    labels = ['MW', 'XLOGP', 'TPSA', 'HBD', 'RotatableBonds']
    values = [row.get(k, 0) for k in labels]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=row['SMILES'][:10] + "..."
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) + 20])),
        showlegend=True,
        title="📊 Molecular Properties Radar Chart"
    )
    return fig

if st.session_state.results is not None:
    st.markdown("### 🧭 Radar Chart for a Molecule")
    idx = st.selectbox(
        "Select a molecule to visualize:",
        st.session_state.results.index,
        format_func=lambda i: st.session_state.results.loc[i, 'SMILES'][:30]
    )
    fig = plot_radar(st.session_state.results.loc[idx])
    st.plotly_chart(fig, use_container_width=True)

    # 🔄 Reset button
    if st.button("🔄 Clear Results & Start New Batch"):
        st.session_state.results = None
        st.session_state.smiles_input = ""
        st.session_state.uploaded_file = None
        st.experimental_rerun()

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 DataBiotica / KinasePred. All rights reserved.")



