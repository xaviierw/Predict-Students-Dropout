import pandas as pd
import streamlit as st

from app_utils import (
    load_artifacts,
    load_defaults,
    default_num,
    gpa_to_grade20,
    build_comparison_df,
    safe_feature_importance_df,
    numeric_cols_only
)

st.set_page_config(
    page_title="EduRisk",
    page_icon="📚",
    layout="wide"
)


def inject_css():
    st.markdown(
        """
        <style>
        /* ================= App background ================= */
        .stApp {
            background: radial-gradient(1200px circle at 0% 0%, rgba(255, 99, 132, 0.10), transparent 55%),
                        radial-gradient(900px circle at 100% 0%, rgba(54, 162, 235, 0.10), transparent 45%),
                        radial-gradient(900px circle at 50% 100%, rgba(255, 206, 86, 0.10), transparent 55%),
                        linear-gradient(180deg, #0b1220 0%, #0b1220 100%);
            color: #e5e7eb;
        }

        /* ================= Text ================= */
        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #e5e7eb !important;
        }

        .muted {
            color: rgba(229,231,235,0.75) !important;
        }

        /* ================= Sidebar ================= */
        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.04);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* ================= Cards ================= */
        .card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 16px 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }

        /* ================= Buttons ================= */
        div.stButton > button {
            border-radius: 12px;
            padding: 10px 14px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.06);
            color: #ffffff;
        }

        div.stButton > button:hover {
            background: rgba(255,255,255,0.12);
        }

        /* ================= Tabs (FLAT UNDERLINE STYLE) ================= */

        /* Tab container spacing + bottom divider */
        div[data-testid="stTabs"] > div {
            border-bottom: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 8px;
        }

        /* All tabs */
        button[data-baseweb="tab"] {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 10px 6px !important;
            margin-right: 22px !important;
            box-shadow: none !important;
            color: rgba(229,231,235,0.75) !important;
            font-weight: 600 !important;
            font-size: 15px !important;
        }

        /* Hover */
        button[data-baseweb="tab"]:hover {
            background: transparent !important;
            color: #ffffff !important;
        }

        /* Active tab text */
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff !important;
        }

        /* Underline indicator (active tab) */
        div[data-baseweb="tab-highlight"] {
            display: block !important;
            height: 2px !important;
            background: #ff4b4b !important;
            border-radius: 999px !important;
        }

        /* ================= Dataframes ================= */
        .stDataFrame, .stTable {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
        }

        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 8px 0;
            font-size: 13px;
            color: rgba(229,231,235,0.6);
            background: linear-gradient(
                to top,
                rgba(11,18,32,0.95),
                rgba(11,18,32,0.70),
                transparent
            );
            z-index: 999;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource
def get_model_and_features():
    return load_artifacts()


@st.cache_data
def get_defaults_df():
    return load_defaults()


def run_prediction(model, feature_list, X_row):
    proba = model.predict_proba(X_row)[0]
    dropout_risk = float(proba[1])
    st.session_state.last_pred = {
        "proba": proba,
        "risk": dropout_risk
    }


def main():
    inject_css()

    model, feature_list = get_model_and_features()
    df_defaults = get_defaults_df()

    # ---------------- Sidebar (controls) ----------------
    st.sidebar.markdown("## Parameters")
    st.sidebar.caption("Adjust inputs, then click **Predict**.")
    st.sidebar.markdown("### Academic Averages")

    ev_min, ev_max, ev_val = default_num(df_defaults, "avg_evaluations", 0, 30, 6)
    avg_evaluations = st.sidebar.slider(
        "Avg assessments attempted",
        min_value=int(ev_min),
        max_value=int(ev_max),
        value=int(ev_val),
        step=1
    )

    ap_min, ap_max, ap_val = default_num(df_defaults, "avg_approved", 0, 30, 5)
    avg_approved = st.sidebar.slider(
        "Avg modules passed",
        min_value=int(ap_min),
        max_value=int(ap_max),
        value=int(ap_val),
        step=1
    )

    st.sidebar.markdown("### GPA → Grade (0–20)")
    gpa1 = st.sidebar.number_input(
        "Semester 1 GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.0,
        step=0.01,
        format="%.2f"
    )

    gpa2 = st.sidebar.number_input(
        "Semester 2 GPA",
        min_value=0.0,
        max_value=4.0,
        value=2.0,
        step=0.01,
        format="%.2f"
    )
    avg_grade = (gpa_to_grade20(gpa1) + gpa_to_grade20(gpa2)) / 2

    st.sidebar.markdown(
        f"""
        <div class="card">
            <div class="muted" style="font-size: 13px;">Current derived value</div>
            <div style="font-size: 30px; font-weight: 800;">{avg_grade:.1f}</div>
            <div class="muted">Average grade (0–20) from GPA</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("### Admin / Finance")
    age = st.sidebar.slider("Age", 15, 70, 20)

    tuition = 1 if st.sidebar.toggle("Tuition fees paid", True) else 0
    debtor = 1 if st.sidebar.toggle("Student in debt", False) else 0
    scholarship = 1 if st.sidebar.toggle("Scholarship holder", False) else 0

    with st.sidebar.expander("Optional: Economic context", expanded=False):
        gdp = st.number_input("GDP", -10.0, 10.0, 1.5)
        inflation = st.number_input("Inflation rate", -10.0, 20.0, 1.0)
        unemployment = st.number_input("Unemployment rate", 0.0, 30.0, 10.0)

    # ---------------- Main header ----------------
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Academic Risk Predictor</h1>
            <div class="muted" style="margin-top: 6px;">Your early intervention tool for institutions. Powered by EduRisk</div>
            <div class="muted">
                <b>Note:</b> This dataset originates from <b>Portugal</b>, where the education system
                does not use GPA as a grading standard. GPA values shown here are an
                approximation used to align the data with Singapore's grading context.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ---------------- Build input row ----------------
    inputs = {
        "avg_approved": avg_approved,
        "avg_grade": avg_grade,
        "avg_evaluations": avg_evaluations,
        "Age": age,
        "Tuition_fees_up_to_date": tuition,
        "Debtor": debtor,
        "Scholarship_holder": scholarship,
        "GDP": gdp,
        "Inflation_rate": inflation,
        "Unemployment_rate": unemployment,
    }

    X_row = pd.DataFrame([[inputs[c] for c in feature_list]], columns=feature_list)

    # ---------------- Predict action (top button) ----------------
    c1, c2, c3 = st.columns([4, 2, 4])

    with c2:
        do_predict_top = st.button("Predict", use_container_width=True, key="predict_top")
        st.markdown(
            '<div class="muted" style="text-align:center; margin-top:6px;">'
            '</div>',
            unsafe_allow_html=True
        )

    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None

    if do_predict_top:
        run_prediction(model, feature_list, X_row)

    st.divider()

    # ---------------- Tabs ----------------
    tab_results, tab_compare, tab_analysis = st.tabs(["Results", "Comparison", "Analysis"])

    last = st.session_state.last_pred

    # ---------------- Results tab ----------------
    with tab_results:
        if last is None:
            st.markdown(
                """
                <div class="card">
                    <div style="font-size: 18px; font-weight: 700;">No prediction yet</div>
                    <div class="muted" style="margin-top: 6px;">
                        Use the sidebar controls and click <b>Predict</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            proba = last["proba"]
            risk = last["risk"]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Dropout Risk", f"{risk:.1%}")
            k2.metric("Not Dropout", f"{float(proba[0]):.1%}")
            k3.metric("Avg Modules Passed", f"{avg_approved}")
            k4.metric("Avg Attempts", f"{avg_evaluations}")

            st.progress(int(risk * 100))

            st.write("")
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("#### Probability breakdown")
                prob_df = pd.DataFrame({
                    "Class": ["Not Dropout", "Dropout"],
                    "Probability": [float(proba[0]), float(proba[1])]
                }).set_index("Class")
                st.bar_chart(prob_df)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("#### Your inputs (summary)")
                summary_df = pd.DataFrame([
                    {"Field": "Age", "Value": age},
                    {"Field": "Avg modules passed", "Value": avg_approved},
                    {"Field": "Avg assessments attempted", "Value": avg_evaluations},
                    {"Field": "Sem 1 GPA", "Value": gpa1},
                    {"Field": "Sem 2 GPA", "Value": gpa2},
                    {"Field": "Avg grade (0–20)", "Value": round(avg_grade, 1)},
                    {"Field": "Tuition up to date", "Value": "Yes" if tuition == 1 else "No"},
                    {"Field": "Debtor", "Value": "Yes" if debtor == 1 else "No"},
                    {"Field": "Scholarship holder", "Value": "Yes" if scholarship == 1 else "No"},
                ])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Comparison tab ----------------
    with tab_compare:
        st.markdown(
            """
            <div class="card">
                <div style="font-size: 18px; font-weight: 700;">You vs Dataset Typical Student</div>
                <div class="muted" style="margin-top: 6px;">
                    This compares your key values against the dataset <b>median</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")

        comp_df = build_comparison_df(df_defaults, {
            "avg_evaluations": avg_evaluations,
            "avg_approved": avg_approved,
            "avg_grade": avg_grade,
            "Age": age
        })

        if comp_df is None or comp_df.empty:
            st.info("Error loading dataset")
        else:
            st.markdown("#### Comparison chart")
            st.bar_chart(comp_df.set_index("Feature"))
            st.markdown("#### Comparison table")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Analysis tab ----------------
    with tab_analysis:
        st.markdown(
            """
            <div class="card">
                <div style="font-size: 18px; font-weight: 700;">Model + Dataset Analysis</div>
                <div class="muted" style="margin-top: 6px;">
                    Feature importance for model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")

        a1 = st.container()

        with a1:
            st.markdown("#### Feature importance (Top 10)")
            imp_df = safe_feature_importance_df(model, feature_list)
            if imp_df is None:
                st.info("This model does not expose feature_importances_.")
            else:
                st.bar_chart(imp_df.set_index("Feature").head(10))
                with st.expander("Full importance table"):
                    st.dataframe(imp_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.markdown(
    """
    <div class="footer">
        © 2026 EduRisk. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
    main()
