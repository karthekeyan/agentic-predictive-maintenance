"""
Streamlit demo UI for the Agentic Predictive Maintenance PoC.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys

sys.path.append('src')
sys.path.append('src/agents')
sys.path.append('src/tools')

from pipeline import build_pipeline

st.set_page_config(page_title="Predictive Maintenance PoC", layout="wide")

# Only reduces default spacing - no colors touched
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("Agentic AI Predictive Maintenance")
st.caption("Validated on real historical failure data — Azure Predictive Maintenance dataset · " +
           "100 machines · 761 real historical failures")


@st.cache_resource
def load_data():
    telemetry = pd.read_csv("data/raw/PdM_telemetry.csv")
    maint = pd.read_csv("data/raw/PdM_maint.csv")
    failures = pd.read_csv("data/raw/PdM_failures.csv")
    machines = pd.read_csv("data/raw/PdM_machines.csv")

    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
    failures['datetime'] = pd.to_datetime(failures['datetime'])
    maint['datetime'] = pd.to_datetime(maint['datetime'])

    from predict_failure_risk import compute_health_scores
    from get_failure_rate_stats import calculate_mtbf_stats

    telemetry_scored = compute_health_scores(telemetry)
    mtbf_table = calculate_mtbf_stats(failures, machines)

    return telemetry_scored, failures, maint, machines, mtbf_table


@st.cache_resource
def load_pipeline():
    return build_pipeline()


with st.spinner("Loading..."):
    telemetry_scored, failures, maint, machines, mtbf_table = load_data()
    pipeline = load_pipeline()

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    machine_id = st.selectbox("Select a machine", sorted(machines['machineID'].unique()))
with col2:
    machine_failures = failures[failures['machineID'] == machine_id]
    if len(machine_failures) > 0:
        selected_failure = st.selectbox(
            "Check risk before this real failure",
            machine_failures['datetime'].tolist(),
            format_func=lambda x: x.strftime('%Y-%m-%d')
        )
    else:
        st.warning("No recorded failures for this machine")
        selected_failure = None
with col3:
    st.write("")  # aligns button vertically with the dropdowns
    run_clicked = st.button("Run diagnosis", type="primary")

if selected_failure is not None and run_clicked:
    model = machines[machines['machineID'] == machine_id]['model'].values[0]
    as_of = selected_failure - pd.Timedelta(hours=12)

    with st.spinner("Running the agent pipeline..."):
        state = {
            'machine_id': machine_id, 'model': model, 'as_of': as_of,
            'telemetry_scored': telemetry_scored, 'failures_df': failures,
            'maint_df': maint, 'machines_df': machines, 'mtbf_table': mtbf_table,
        }
        result = pipeline.invoke(state)

    health = result['health_result']
    diagnosis = result['diagnosis_result']
    recommendation = result['recommendation_result']
    routing = result['routing_result']

    c1, c2, c3 = st.columns(3)
    c1.metric("Health score", f"{health['health_score']:.3f}", health['risk_level'])
    c2.metric("Diagnosis", diagnosis['likely_component'] or "N/A",
              f"{diagnosis['confidence']}% confidence" if diagnosis['confidence'] else "no data")
    c3.metric("Routing", routing['routing_decision'])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Reasoning**")
        st.write(diagnosis['reasoning'])
    with col_b:
        st.markdown("**Recommendation**")
        st.write(recommendation['recommendation'])

    st.caption(f"Actual real failure on this date: {selected_failure.date()} — compare against the diagnosis above")

    if routing['routing_decision'] != 'auto_route':
        st.warning("This case requires human engineer review before proceeding.")
    else:
        st.info("High confidence — auto-routed to maintenance team.")