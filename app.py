# ==========================================================
# app.py — 2026 eThekwini Election Forecast Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="2026 Election Forecast — eThekwini",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗳️ 2026 South African Local Government Election Forecast")
st.subheader("eThekwini Metropolitan Municipality")
st.caption("Analytical model output · Not a political prediction · Election date: 04 November 2026")

st.divider()

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "election_master.csv")
MODEL_PKL = os.path.join(BASE_DIR, "election_2026_forecast_model.pkl")

REGISTERED_2024 = 1989823
PROJECTED_TURNOUT_2026 = 0.4890
PROJECTED_VOTES_2026 = int(PROJECTED_TURNOUT_2026 * REGISTERED_2024)

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
@st.cache_data
def load_master():
    if not os.path.exists(MASTER_CSV):
        return None
    return pd.read_csv(MASTER_CSV)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PKL):
        return None
    return joblib.load(MODEL_PKL)

df_model = load_master()

if df_model is None:
    st.error(
        "⚠️ `election_master.csv` not found in this folder.\n\n"
        "**Fix:** Open `MunicipalVotes (2).ipynb`, scroll to the bottom, add:\n\n"
        "```python\n"
        "df_model.to_csv('election_master.csv', index=False)\n"
        "```\n\n"
        "Then re-run that cell and refresh this page."
    )
    st.stop()

# Try loading the model (optional — only used if we need to recompute)
model_data = load_model()
rf_model = model_data["model"] if model_data else None

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

top_n = st.sidebar.slider(
    "Show top N parties",
    min_value=5, max_value=30, value=10, step=1
)

min_share = st.sidebar.slider(
    "Minimum vote share filter (%)",
    min_value=0.0, max_value=5.0, value=0.1, step=0.1
)

st.sidebar.divider()
st.sidebar.markdown("### 📁 Data Sources")
st.sidebar.markdown(
    "- **2024 National & Provincial** — IEC\n"
    "- **2021 Local Government** — IEC\n"
    "- **2016 Local Government** — IEC\n"
    "- **Ward Boundaries** — MDB"
)

st.sidebar.divider()
st.sidebar.warning(
    "**Disclaimer:** All 2026 figures are model estimates with inherent "
    "uncertainty. The 2024 National election is not directly comparable "
    "to a Local Government election."
)

# ----------------------------------------------------------
# Filter
# ----------------------------------------------------------
df_filtered = df_model[
    (df_model["Pct_2016"] >= min_share / 100) |
    (df_model["Pct_2021"] >= min_share / 100) |
    (df_model["Pct_2024"] >= min_share / 100)
].copy()

# ----------------------------------------------------------
# KPIs
# ----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Registered Voters", f"{REGISTERED_2024:,}")

with c2:
    st.metric("Projected 2026 Turnout", f"{PROJECTED_TURNOUT_2026:.2%}")

with c3:
    st.metric("Projected 2026 Votes Cast", f"{PROJECTED_VOTES_2026:,}")

with c4:
    if "Pct_2026_Predicted" in df_model.columns:
        leader = df_model.sort_values("Pct_2026_Predicted", ascending=False).iloc[0]
        st.metric(
            "Leading Party (projected)",
            str(leader["Party_Std"]),
            f"{leader['Pct_2026_Predicted'] * 100:.1f}%"
        )
    else:
        st.metric("Leading Party (projected)", "N/A")

st.divider()

# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Historical Vote Share",
    "🔮 2026 Projection",
    "📈 Turnout",
    "📋 Full Data"
])

# ==========================================================
# TAB 1 — Historical
# ==========================================================
with tab1:
    st.header("Historical Vote Share — eThekwini")

    year_choice = st.radio(
        "Select year:",
        options=["2024", "2021", "2016"],
        horizontal=True
    )

    df_sorted = df_filtered.sort_values(
        f"Pct_{year_choice}", ascending=False
    ).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        df_sorted["Party_Std"],
        df_sorted[f"Pct_{year_choice}"] * 100,
        color="steelblue",
        edgecolor="black"
    )
    ax.set_title(f"{year_choice} Vote Share — eThekwini (Top {top_n})")
    ax.set_xlabel("Party")
    ax.set_ylabel("Vote Share (%)")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, df_sorted[f"Pct_{year_choice}"] * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("Three-Year Comparison")

    top_compare = df_filtered.sort_values(
        "Pct_2024", ascending=False
    ).head(top_n)

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(top_compare))
    w = 0.27

    ax2.bar(x - w, top_compare["Pct_2016"] * 100, w,
            label="2016", color="steelblue", edgecolor="black")
    ax2.bar(x,     top_compare["Pct_2021"] * 100, w,
            label="2021", color="orange", edgecolor="black")
    ax2.bar(x + w, top_compare["Pct_2024"] * 100, w,
            label="2024", color="green", edgecolor="black")

    ax2.set_xticks(x)
    ax2.set_xticklabels(top_compare["Party_Std"], rotation=45, ha="right")
    ax2.set_ylabel("Vote Share (%)")
    ax2.set_title("Historical Vote Share Comparison — eThekwini")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)

# ==========================================================
# TAB 2 — 2026 Projection
# ==========================================================
with tab2:
    st.header("Projected 2026 Vote Share")

    if "Pct_2026_Predicted" not in df_model.columns:
        st.warning(
            "No 2026 projections available in `election_master.csv`. "
            "Re-run the notebook's last cells to regenerate."
        )
    else:
        df_2026 = df_model.sort_values(
            "Pct_2026_Predicted", ascending=False
        ).head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(
            df_2026["Party_Std"],
            df_2026["Pct_2026_Predicted"] * 100,
            color="crimson",
            edgecolor="black"
        )
        ax.set_title("Projected 2026 Vote Share — eThekwini (Model Estimate)")
        ax.set_xlabel("Party")
        ax.set_ylabel("Projected Vote Share (%)")
        plt.xticks(rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, df_2026["Pct_2026_Predicted"] * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}%",
                ha="center", va="bottom", fontsize=8
            )

        plt.tight_layout()
        st.pyplot(fig)

        # Coalition scenario
        st.divider()
        st.subheader("🗳️ Coalition Scenario Analysis")

        df_coal = df_model.sort_values(
            "Pct_2026_Predicted", ascending=False
        ).reset_index(drop=True)

        majority = df_coal[df_coal["Pct_2026_Predicted"] > 0.50]

        if len(majority) > 0:
            st.success(
                f"**Outright majority projected for "
                f"{majority.iloc[0]['Party_Std']}** "
                f"({majority.iloc[0]['Pct_2026_Predicted'] * 100:.1f}%)"
            )
        else:
            st.warning("No outright majority projected. A coalition would be required.")
            cumulative = 0
            coalition = []
            for _, row in df_coal.iterrows():
                coalition.append(
                    (row["Party_Std"], row["Pct_2026_Predicted"])
                )
                cumulative += row["Pct_2026_Predicted"]
                if cumulative > 0.50:
                    break

            st.markdown("**Minimal winning coalition (top-down):**")
            for party, pct in coalition:
                st.write(f"- {party}: **{pct * 100:.2f}%**")
            st.write(f"**Combined share:** {cumulative * 100:.2f}%")

        # Projected votes
        st.divider()
        st.subheader("Projected 2026 Vote Totals")

        df_2026_v = df_2026.copy()
        df_2026_v["Projected_Votes"] = (
            df_2026_v["Pct_2026_Predicted"] * PROJECTED_VOTES_2026
        ).round(0).astype(int)

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        bars3 = ax3.bar(
            df_2026_v["Party_Std"],
            df_2026_v["Projected_Votes"],
            color="teal",
            edgecolor="black"
        )
        ax3.set_title("Projected 2026 Vote Totals — eThekwini")
        ax3.set_xlabel("Party")
        ax3.set_ylabel("Projected Votes")
        plt.xticks(rotation=45, ha="right")
        ax3.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars3, df_2026_v["Projected_Votes"]):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(value):,}",
                ha="center", va="bottom", fontsize=7
            )

        plt.tight_layout()
        st.pyplot(fig3)

# ==========================================================
# TAB 3 — Turnout
# ==========================================================
with tab3:
    st.header("Voter Turnout Analysis")

    total_2016 = df_model["Votes_2016"].sum() if "Votes_2016" in df_model.columns else 0
    total_2021 = df_model["Votes_2021"].sum() if "Votes_2021" in df_model.columns else 0
    total_2024 = df_model["Votes_2024"].sum() if "Votes_2024" in df_model.columns else 0

    t2016 = total_2016 / REGISTERED_2024 if REGISTERED_2024 else 0
    t2021 = total_2021 / REGISTERED_2024 if REGISTERED_2024 else 0
    t2024 = total_2024 / REGISTERED_2024 if REGISTERED_2024 else 0
    t2026 = PROJECTED_TURNOUT_2026

    years = ["2016", "2021", "2024", "2026 (proj.)"]
    values = [t2016 * 100, t2021 * 100, t2024 * 100, t2026 * 100]
    colors = ["steelblue", "steelblue", "steelblue", "crimson"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(years, values, color=colors, edgecolor="black")
    ax.set_title("Turnout Trend and 2026 Projection — eThekwini")
    ax.set_ylabel("Turnout Rate (%)")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center", va="bottom"
        )

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.markdown("### Turnout Metrics")
    cols = st.columns(4)
    labels = ["2016", "2021", "2024", "2026 (projected)"]
    vals = [t2016, t2021, t2024, t2026]
    for i, (lbl, val) in enumerate(zip(labels, vals)):
        with cols[i]:
            st.metric(lbl, f"{val * 100:.2f}%")

    st.info(
        "**Assumption:** Registered population held at the 2024 baseline "
        f"({REGISTERED_2024:,}). Projection uses a simple linear trend "
        "on 2016, 2021, 2024 turnout. Model estimate, not certainty."
    )

# ==========================================================
# TAB 4 — Full Data
# ==========================================================
with tab4:
    st.header("Full Data Table")

    display_cols = ["Party_Std", "Pct_2016", "Pct_2021", "Pct_2024"]
    if "Pct_2026_Predicted" in df_model.columns:
        display_cols.append("Pct_2026_Predicted")

    out = df_model[display_cols].copy()
    for c in display_cols[1:]:
        out[c] = (out[c] * 100).round(3)

    out = out.rename(columns={
        "Party_Std": "Party",
        "Pct_2016": "2016 (%)",
        "Pct_2021": "2021 (%)",
        "Pct_2024": "2024 (%)",
        "Pct_2026_Predicted": "2026 Projected (%)"
    })

    sort_col = "2026 Projected (%)" if "2026 Projected (%)" in out.columns else "2024 (%)"
    out = out.sort_values(sort_col, ascending=False)

    st.dataframe(out, use_container_width=True, height=600)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name="eThekwini_2026_forecast.csv",
        mime="text/csv"
    )

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------
st.divider()
st.caption(
    "**Sources:** IEC South Africa (results.elections.org.za), "
    "Municipal Demarcation Board. "
    "**Model:** Random Forest Regressor on historical vote shares. "
    "**Disclaimer:** All 2026 values are model estimates."
)