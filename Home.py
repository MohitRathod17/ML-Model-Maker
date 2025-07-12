import streamlit as st

st.set_page_config(page_title="Machine Learning Apps", layout="centered")

st.title("📂 ML App Hub")
st.markdown("---")

st.markdown("Welcome to the **Streamlit Multi-Tool App Suite**. Select a module below to get started:")

st.markdown("### 🧰 Available Modules")

# Null Value Filler
st.markdown("#### 🔢 [Null Value Filler](http://localhost:8501/NullValueFiller)")
st.markdown("- Detect and fill missing values using different strategies like mean, median, mode, random, etc.")

# Encoder
st.markdown("#### 🧬 [Encoder](http://localhost:8501/encoder)")
st.markdown("- One-hot encode or label encode categorical features.")

# Data Visualizer
st.markdown("#### 📊 [Data Visualizer](http://localhost:8501/dataviz)")
st.markdown("- Upload CSV files and explore distributions, correlations, and custom plots interactively.")

# Supervised Model Maker
st.markdown("#### 📊 [Supervised Model Maker](http://localhost:8501/Supervised_Model_Maker)")
st.markdown("- Upload CSV files and make a perfect tuned Supervised ML model for your data.")

# Un-Supervised Model Maker
st.markdown("#### 🔍 [Un-Supervised Model Maker](http://localhost:8501/unsupervised)")
st.markdown("- Upload CSV files and make a perfect tuned Unsupervised ML model for your data.")

# DL Model Maker
st.markdown("#### 🧠 [Deep Learning Model Maker](http://localhost:8501/deep_learning)")
st.markdown("- Upload CSV files and make a perfect Neural Network model for your data.")

st.markdown("---")
