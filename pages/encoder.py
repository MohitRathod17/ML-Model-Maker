import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from io import BytesIO
import numpy as np

st.set_page_config(page_title="🧬 Object Encoder", layout="wide")
st.title("🧬 Encode Object Columns in CSV")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not object_cols:
        st.warning("No object or categorical columns found in the dataset.")
    else:
        st.success(f"Detected categorical columns: {object_cols}")

        selected_cols = st.multiselect("Select columns to encode:", object_cols, default=object_cols)
        encoding_type = st.radio("Choose Encoding Method:", ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"])

        if encoding_type == "Label Encoding":
            df_encoded = df.copy()
            for col in selected_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            st.success("Label Encoding applied successfully.")

        elif encoding_type == "One-Hot Encoding":
            df_encoded = pd.get_dummies(df, columns=selected_cols)
            st.success("One-Hot Encoding applied successfully.")

        elif encoding_type == "Ordinal Encoding":
            df_encoded = df.copy()
            oe = OrdinalEncoder()
            df_encoded[selected_cols] = oe.fit_transform(df_encoded[selected_cols].astype(str))
            st.success("Ordinal Encoding applied successfully.")

        st.subheader("🔍 Encoded Data Preview")
        st.dataframe(df_encoded.head())

        # Download encoded CSV
        buffer = BytesIO()
        df_encoded.to_csv(buffer, index=False)
        st.download_button(
            label="📥 Download Encoded CSV",
            data=buffer.getvalue(),
            file_name="encoded_data.csv",
            mime="text/csv"
        )
