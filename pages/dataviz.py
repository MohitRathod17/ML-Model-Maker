import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CSV Data Visualizer", layout="wide")
st.title("📊 CSV Data Visualization App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df_original = pd.read_csv(uploaded_file)

        # Fix Arrow serialization issues
        for col in df_original.columns:
            if pd.api.types.is_integer_dtype(df_original[col]):
                df_original[col] = pd.to_numeric(df_original[col], downcast="integer", errors='coerce')
            elif pd.api.types.is_float_dtype(df_original[col]):
                df_original[col] = pd.to_numeric(df_original[col], errors='coerce')
            elif pd.api.types.is_object_dtype(df_original[col]):
                df_original[col] = df_original[col].astype(str)

        st.success("✅ CSV file loaded successfully!")

        # Limit rows
        st.subheader("📉 Limit Data by Percentage")
        percent = st.slider("Select percentage of rows to use", 1, 100, 100)
        limit = max(1, int(len(df_original) * (percent / 100.0)))
        df = df_original.head(limit)

        st.info(f"Using top {limit:,} rows out of total {len(df_original):,} rows.")

        # Show data
        st.subheader("🔍 Data Preview")
        st.dataframe(df)

        # Summary statistics
        st.subheader("📈 Data Summary")
        st.write(df.describe(include='all'))

        # Column names
        st.subheader("🧲 Column Names")
        st.write(df.columns.tolist())

        # Visualization section
        st.subheader("📊 Data Visualization")
        numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()

        if len(numeric_cols) >= 1:
            chart_type = st.selectbox("Select Chart Type", [
                "Line", "Bar", "Scatter", "Histogram", "Box Plot", "Pie Chart", "Heatmap"])

            if chart_type == "Line":
                x_col = st.selectbox("X-axis column", df.columns)
                y_col = st.selectbox("Y-axis column (numeric)", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)

            elif chart_type == "Bar":
                x_col = st.selectbox("X-axis column", df.columns)
                y_col = st.selectbox("Y-axis column (numeric)", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)

            elif chart_type == "Scatter":
                x_col = st.selectbox("X-axis column (numeric)", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis column (numeric)", numeric_cols, key="scatter_y")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)

            elif chart_type == "Histogram":
                col = st.selectbox("Select column (numeric)", numeric_cols, key="hist_col")
                bins = st.slider("Number of bins", 5, 100, 20)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(df[col], bins=bins, kde=True, ax=ax)
                st.pyplot(fig)

            elif chart_type == "Box Plot":
                col = st.selectbox("Select column (numeric)", numeric_cols, key="box_col")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(y=df[col], ax=ax)
                st.pyplot(fig)

            elif chart_type == "Pie Chart":
                cat_col = st.selectbox("Select column (categorical)", df.select_dtypes(include='object').columns)
                value_counts = df[cat_col].value_counts().head(10)
                fig, ax = plt.subplots()
                ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

            elif chart_type == "Heatmap":
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

        else:
            st.warning("❗ No numeric columns available for visualization.")

    except Exception as e:
        st.error(f"❌ Error loading CSV file: {e}")

else:
    st.warning("📂 Please upload a CSV file to begin.")
