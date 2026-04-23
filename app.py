import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.mitigation import Reweighing

st.set_page_config(page_title="FairGuard AI", layout="wide", page_icon="🛡️")
st.title("🛡️ FairGuard AI")
st.markdown("**Detect • Measure • Fix** Hidden Bias in AI Decisions")

# Sidebar
page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Bias Scan", "Mitigation", "Generate Report"])

# Session state
for key in ['df', 'metrics_before', 'metrics_after', 'accuracy_before', 'accuracy_after', 
            'protected_attr', 'y_test', 'y_pred_before', 'group_test']:
    if key not in st.session_state:
        st.session_state[key] = None

# ==================== HOME ====================
if page == "Home":
    st.header("Welcome to FairGuard AI")
    if st.button("📊 Load Adult Income Dataset"):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        col_names = ["age", "workclass", "fnlwgt", "education", "education_num", 
                    "marital_status", "occupation", "relationship", "race", "sex", 
                    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
        
        df = pd.read_csv(url, names=col_names, skipinitialspace=True, na_values='?')
        df = df.dropna().reset_index(drop=True)
        st.session_state['df'] = df
        st.success("Adult dataset loaded and cleaned!")
        st.dataframe(df.head())

# ==================== UPLOAD ====================
elif page == "Upload Data":
    st.header("Upload Your Own Dataset")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, na_values='?')
        df = df.dropna().reset_index(drop=True)
        st.session_state['df'] = df
        st.success(f"Loaded and cleaned: {df.shape[0]:,} rows")
        st.dataframe(df.head())

# ==================== BIAS SCAN ====================
elif page == "Bias Scan":
    st.header(" Bias Scan")
    if st.session_state['df'] is None:
        st.warning("Load a dataset first.")
        st.stop()

    df = st.session_state['df'].copy()

    target_col = st.selectbox("Target Column", df.columns, index=len(df.columns)-1)
    protected_attrs = st.multiselect(
        "Protected Attributes",
        [col for col in df.columns if col != target_col],
        default=["sex", "race"] if "sex" in df.columns else None
    )

    if st.button("🚀 Run Bias Scan", type="primary"):
        with st.spinner("Training model and scanning bias..."):
            X = df.drop(columns=[target_col])
            y_raw = df[target_col].astype(str).str.strip()

            le_y = LabelEncoder()
            y_encoded = le_y.fit_transform(y_raw)
            if len(np.unique(y_encoded)) > 2:
                st.error(f"Target '{target_col}' has more than 2 classes. Only binary targets supported.")
                st.stop()
            
            y = (y_encoded == 1).astype(int)

            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = (model.predict(X_test) == 1).astype(int)

            acc = accuracy_score(y_test, y_pred)
            st.session_state['accuracy_before'] = acc
            st.session_state['y_test'] = y_test
            st.session_state['y_pred_before'] = y_pred

            prot_attr = protected_attrs[0]
            st.session_state['protected_attr'] = prot_attr
            group = df.loc[X_test.index, prot_attr].reset_index(drop=True)

            if group.nunique() <= 2:
                le_g = LabelEncoder()
                g_enc = le_g.fit_transform(group)
                group_a = (g_enc == 0).astype(int)
                group_b = (g_enc == 1).astype(int)
            else:
                median = group.median()
                group_a = (group <= median).astype(int)
                group_b = (group > median).astype(int)

            metrics_before = classification_bias_metrics(
                group_a=group_a,
                group_b=group_b,
                y_pred=y_pred,
                y_true=y_test,
                metric_type='both'
            )

            st.session_state['metrics_before'] = metrics_before
            st.session_state['group_test'] = group

            st.subheader(f"Bias Metrics — **{prot_attr}** (Before)")

            def color_metric(val):
                if isinstance(val, (int, float)):
                    if abs(val) < 0.05:
                        return 'background-color: lightgreen'
                    elif abs(val) < 0.15:
                        return 'background-color: lightyellow'
                    else:
                        return 'background-color: lightcoral'
                return ''

            styled_metrics = metrics_before.style.map(color_metric)
            st.dataframe(styled_metrics, use_container_width=True)

            test_df = pd.DataFrame({'group': group, 'prediction': y_pred})
            fig = px.bar(test_df.groupby('group')['prediction'].mean().reset_index(), 
                         x='group', y='prediction', 
                         title=f"Positive Prediction Rate by {prot_attr} (Before)")
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"Baseline Accuracy: **{acc:.2%}**")

# ==================== MITIGATION ====================
elif page == "Mitigation":
    st.header("⚖️ Bias Mitigation")
    if st.session_state.get('metrics_before') is None:
        st.warning("Run Bias Scan first!")
        st.stop()

    if st.button("🔧 Apply Reweighing Mitigation", type="primary"):
        with st.spinner("Applying Reweighing..."):
            df = st.session_state['df'].copy()
            target_col = list(df.columns)[-1]
            prot_attr = st.session_state['protected_attr']

            X = df.drop(columns=[target_col])
            y_raw = df[target_col].astype(str).str.strip()

            le_y = LabelEncoder()
            y_encoded = le_y.fit_transform(y_raw)
            y = (y_encoded == 1).astype(int)

            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            group_train = df.loc[X_train.index, prot_attr].reset_index(drop=True)
            if group_train.nunique() <= 2:
                le_g = LabelEncoder()
                g_enc = le_g.fit_transform(group_train)
                group_a_train = (g_enc == 0).astype(int)
                group_b_train = (g_enc == 1).astype(int)
            else:
                median = group_train.median()
                group_a_train = (group_train <= median).astype(int)
                group_b_train = (group_train > median).astype(int)

            reweigher = Reweighing()
            reweigher.fit(y=y_train, group_a=group_a_train, group_b=group_b_train)
            sample_weight = reweigher.estimator_params.get("sample_weight")

            model_mit = LogisticRegression(max_iter=1000)
            model_mit.fit(X_train, y_train, sample_weight=sample_weight)

            y_pred_mit = (model_mit.predict(X_test) == 1).astype(int)
            acc_after = accuracy_score(y_test, y_pred_mit)
            st.session_state['accuracy_after'] = acc_after

            group_test = df.loc[X_test.index, prot_attr].reset_index(drop=True)
            if group_test.nunique() <= 2:
                le_g = LabelEncoder()
                g_enc = le_g.fit_transform(group_test)
                group_a_test = (g_enc == 0).astype(int)
                group_b_test = (g_enc == 1).astype(int)
            else:
                median = group_test.median()
                group_a_test = (group_test <= median).astype(int)
                group_b_test = (group_test > median).astype(int)

            metrics_after = classification_bias_metrics(
                group_a=group_a_test,
                group_b=group_b_test,
                y_pred=y_pred_mit,
                y_true=y_test,
                metric_type='both'
            )
            st.session_state['metrics_after'] = metrics_after

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Before")
                st.metric("Accuracy", f"{st.session_state['accuracy_before']:.2%}")
                st.dataframe(st.session_state['metrics_before'])
            with col2:
                st.subheader("After")
                st.metric("Accuracy", f"{acc_after:.2%}", 
                          delta=f"{acc_after - st.session_state['accuracy_before']:.2%}")
                styled_after = metrics_after.style.map(color_metric) if 'color_metric' in locals() else metrics_after
                st.dataframe(styled_after, use_container_width=True)

            comparison = pd.DataFrame({
                'Group': group_test,
                'Before': st.session_state['y_pred_before'],
                'After': y_pred_mit
            }).groupby('Group').mean().reset_index()

            fig = px.bar(comparison, x='Group', y=['Before', 'After'], barmode='group',
                         title="Prediction Rate: Before vs After")
            st.plotly_chart(fig, use_container_width=True)

            st.success("Mitigation applied successfully!")

# ==================== REPORT ====================
elif page == "Generate Report":
    st.header("Generate Report")
    st.info("PDF report with LLM summary coming will be implemented in future updates.")

st.sidebar.caption("FairGuard AI - Pandas Styler Fixed (map instead of applymap)")