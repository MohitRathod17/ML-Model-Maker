import streamlit as st
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="Null Value Filler", layout="wide")
st.title("🧪 Null Value Filler App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Copy for filling
    filled_df = df.copy()

    # Columns with nulls
    null_cols = df.columns[df.isnull().any()].tolist()

    if null_cols:
        st.subheader("Null Columns Detected:")
        st.write(null_cols)

        st.subheader("Choose Fill Methods for Null Columns")

        fill_options = {}
        for col in null_cols:
            col_type = df[col].dtype
            if np.issubdtype(col_type, np.number):
                options = ["Mean", "Median", "Mode", "Min", "Max", "Zero", "Random"]
            else:
                options = ["Mode", "Empty String", "Random"]
            fill_options[col] = st.selectbox(f"Choose fill method for '{col}'", options)

        if st.button("Fill Null Values"):
            for col in null_cols:
                method = fill_options[col]
                non_null_values = df[col].dropna()

                if method == "Mean":
                    value = non_null_values.mean()
                elif method == "Median":
                    value = non_null_values.median()
                elif method == "Mode":
                    value = non_null_values.mode().iloc[0]
                elif method == "Min":
                    value = non_null_values.min()
                elif method == "Max":
                    value = non_null_values.max()
                elif method == "Zero":
                    value = 0
                elif method == "Empty String":
                    value = ""
                elif method == "Random":
                    if not non_null_values.empty:
                        value = random.choice(non_null_values.tolist())
                    else:
                        value = np.nan  # fallback
                else:
                    value = None

                filled_df[col] = df[col].fillna(value)

            st.subheader("✅ Data After Filling Null Values")
            st.dataframe(filled_df)

            # Download option
            csv = filled_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Filled CSV",
                data=csv,
                file_name="filled_data.csv",
                mime="text/csv",
            )
    else:
        st.success("🎉 No null values found in the uploaded dataset!")
else:
    st.info("📁 Please upload a CSV file to begin.")
