import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
import io

st.set_page_config(page_title="🧠 Custom Deep Learning Builder", layout="wide")
st.title("🧠 Deep Learning Neural Network Builder")

# File upload
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select Target Column", df.columns)
    features = st.multiselect("🧠 Select Feature Columns", [col for col in df.columns if col != target])

    if features and target:
        X = df[features]
        y = df[target]

        # Task type selector
        task_type = st.radio("🤖 Select the Task Type", ["Classification", "Regression"])

        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task_type == "Classification":
            if y.dtype != "int":
                y = pd.factorize(y)[0]
            y_encoded = to_categorical(y)
            output_dim = y_encoded.shape[1]
        else:
            y_encoded = y.values.reshape(-1, 1)
            output_dim = 1

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        # Model Architecture
        st.subheader("🛠️ Build Your Neural Network")
        num_layers = st.number_input("🔢 Number of Hidden Layers", min_value=1, max_value=10, value=2)

        layers = []
        for i in range(num_layers):
            st.markdown(f"**Layer {i+1} Configuration**")
            units = st.slider(f"Neurons in Layer {i+1}", 4, 256, 64, key=f"units_{i}")
            activation = st.selectbox(
                f"Activation Function for Layer {i+1}",
                ["relu", "sigmoid", "tanh", "softmax", "linear"],
                key=f"activation_{i}"
            )
            layers.append((units, activation))

        # Compile options
        optimizer = st.selectbox("⚙️ Optimizer", ["adam", "sgd", "rmsprop"])
        epochs = st.slider("🕒 Training Epochs", 10, 1000, 100)
        batch_size = st.slider("📦 Batch Size", 8, 128, 32)

        if st.button("🚀 Train Neural Network"):
            model = Sequential()
            model.add(Dense(layers[0][0], activation=layers[0][1], input_shape=(X_train.shape[1],)))
            for units, activation in layers[1:]:
                model.add(Dense(units, activation=activation))
            model.add(Dense(output_dim, activation="softmax" if task_type == "Classification" else "linear"))

            loss = "categorical_crossentropy" if task_type == "Classification" else "mse"
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy" if task_type == "Classification" else "mse"])

            # Train
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            # Predict & Evaluate
            y_pred = model.predict(X_test)
            st.subheader("📈 Model Evaluation")

            if task_type == "Classification":
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true_class = np.argmax(y_test, axis=1)
                acc = accuracy_score(y_true_class, y_pred_class)
                st.write(f"✅ Accuracy: {acc:.4f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"📉 Mean Squared Error: {mse:.4f}")
                st.write(f"📊 R² Score: {r2:.4f}")

            # Save model
            buffer = io.BytesIO()
            model.save("custom_model.h5")
            with open("custom_model.h5", "rb") as f:
                st.download_button("📥 Download Model (.h5)", f, file_name="custom_model.h5")
else:
    st.info("📤 Upload your dataset to begin.")
