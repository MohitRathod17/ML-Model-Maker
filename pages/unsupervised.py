import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans , DBSCAN , AgglomerativeClustering
import io

st.set_page_config(page_title="Unsupervised Model Trainer", layout="wide")
st.title("🔍 Unsupervised Learning App")

uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Data Preview")
    st.dataframe(df.head())

    st.markdown("### ⚙️ Select Feature Columns for Unsupervised Learning")
    features = st.multiselect("Select features:", df.columns)

    if features:
        X = df[features]
        st.markdown("### 🔧 Choose Unsupervised Learning Type")
        unsupervised_type = st.radio("Unsupervised Learning Type", ["Clustering", "Dimensionality Reduction"])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if unsupervised_type == "Clustering":
            algo = st.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative Clustering"])
            model = None

            if algo == "KMeans":
                n_clusters = st.slider("Enter number of clusters", 2 , 40 , 1)
                model = KMeans(n_clusters=n_clusters)
            elif algo == "DBSCAN":
                eps = st.slider("Enter epsilon", 0.1 , 10.0 , 0.5)
                min_samples = st.slider("Enter min samples", 1 , 20 , 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif algo == "Agglomerative Clustering":
                n_clusters = st.slider("Enter number of clusters", 2 , 40 , 1)
                model = AgglomerativeClustering(n_clusters=n_clusters)
            
            if st.button("Train Clustering Model"):
                labels = model.fit_predict(X_scaled)
                df["Cluster"] = labels
                st.subheader("📊 Clustering Model Performance")
                st.dataframe(df)

                st.markdown("### 📊 Visualize Clusters")
                if X_scaled.shape[1] > 2:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(X_scaled)
                else:
                    reduced_data = X_scaled

                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="Set2")
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                st.pyplot(plt)

                output = io.BytesIO()
                joblib.dump(model, output)
                st.download_button("📥 Download Model", data=output.getvalue(), file_name="unsupervised_model.pkl")

        elif unsupervised_type == "Dimensionality Reduction":
            algo = st.selectbox("Choose Dimensionality Reduction Algorithm", ["PCA", "t-SNE"])
            reduced = None

            if algo == "PCA":
                n_components = st.slider("Number of Components", 2, min(len(features), 10), 2)
                model = PCA(n_components=n_components)
                reduced = model.fit_transform(X_scaled)
            elif algo == "t-SNE":
                n_components = st.slider("Number of Components", 2, 3, 2)
                perplexity = st.slider("Perplexity", 5, 50, 30)
                model = TSNE(n_components=n_components, perplexity=perplexity)
                reduced = model.fit_transform(X_scaled)

            if reduced is not None:
                st.subheader("📉 Reduced Data Preview")
                st.dataframe(pd.DataFrame(reduced, columns=[f"Component {i+1}" for i in range(reduced.shape[1])]))

                st.markdown("### 📊 Visualization")
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1])
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                st.pyplot(plt)

                output = io.BytesIO()
                joblib.dump(model, output)
                st.download_button("📥 Download Dimensionality Reducer", data=output.getvalue(), file_name="dim_reducer.pkl")

else:
    st.info("Please upload a CSV file to start.")