# 🧠 ML Model Maker

A powerful Streamlit-based application that lets you explore data, handle preprocessing, train machine learning models (supervised, unsupervised, and neural networks), tune hyperparameters, and save trained models – all from an interactive UI!

## 🚀 Features

- 📁 Upload your `.csv` dataset
- 📊 Visualize and explore data (histograms, correlation heatmaps, distributions)
- 🧼 Fill missing values (mean, median, drop, etc.)
- 🔤 Encode categorical features (Label Encoding / One-Hot)
- 🧠 Train machine learning models:
  - Supervised (Classification and Regression)
  - Unsupervised (Clustering, Dimensionality Reduction)
  - Neural Networks (fully customizable architecture)
- 🎛 Hyperparameter tuning with UI sliders/selectors
- 💾 Save/export trained models for reuse
- 📈 View evaluation metrics and plots

## 🖼 App Preview

![Demo](demo.gif)

## 📦 Installation

```bash
git clone https://github.com/your-username/ml-model-maker.git
cd ml-model-maker
pip install -r requirements.txt
streamlit run Home.py
