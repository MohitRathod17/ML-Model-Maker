import streamlit as st
import pandas as pd
import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="Supervised Model Trainer", layout="wide")
st.title("📊 Supervised Machine Learning Trainer")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select the target column:", df.columns)
    features = st.multiselect("🧠 Select feature columns:", [col for col in df.columns if col != target])

    if target and features:
        X = df[features]
        y = df[target]

        task_type = st.radio("🧪 Select the ML Task Type:", ["Classification", "Regression"])

        classification_models = [
            "Logistic Regression",
            "Random Forest Classifier",
            "Support Vector Classifier",
            "k-Nearest Neighbors",
            "Decision Tree",
            "Voting Classifier (Combine Models)"
        ]

        regression_models = [
            "Linear Regression",
            "Random Forest Regressor",
            "Support Vector Regressor",
            "k-Nearest Neighbors Regressor",
            "Decision Tree Regressor"
        ]

        model_type = st.selectbox("🧰 Choose a model:", classification_models if task_type == "Classification" else regression_models)
        model = None

        # Classification Models
        if model_type == "Logistic Regression":
            c_val = st.slider("C (Inverse of regularization strength)", 0.01, 10.0, 1.0)
            max_iter = st.slider("Max Iterations", 100, 2000, 1000)
            model = LogisticRegression(C=c_val, max_iter=max_iter)

        elif model_type == "Random Forest Classifier":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 50, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        elif model_type == "Support Vector Classifier":
            C = st.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = SVC(C=C, kernel=kernel)

        elif model_type == "k-Nearest Neighbors":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        elif model_type == "Decision Tree":
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            max_depth = st.slider("Max Depth", 1, 50, 5)
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

        elif model_type == "Voting Classifier (Combine Models)":
            clf1 = LogisticRegression()
            clf2 = RandomForestClassifier()
            clf3 = SVC(probability=True)
            model = VotingClassifier(estimators=[("lr", clf1), ("rf", clf2), ("svc", clf3)], voting='soft')

        # Regression Models
        elif model_type == "Linear Regression":
            model = LinearRegression()

        elif model_type == "Random Forest Regressor":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 50, 10)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

        elif model_type == "Support Vector Regressor":
            C = st.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = SVR(C=C, kernel=kernel)

        elif model_type == "k-Nearest Neighbors Regressor":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)

        elif model_type == "Decision Tree Regressor":
            max_depth = st.slider("Max Depth", 1, 50, 10)
            model = DecisionTreeRegressor(max_depth=max_depth)

        test_size = st.slider("📏 Select Test Size (Split Ratio):", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if st.button("🚀 Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📈 Model Performance")
            if task_type == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"📉 Mean Squared Error: {mse:.4f}")
                st.write(f"📊 R² Score: {r2:.4f}")
            else:
                acc = accuracy_score(y_test, y_pred)
                st.write(f"✅ Accuracy Score: {acc:.4f}")
                st.text("📋 Classification Report:")
                st.text(classification_report(y_test, y_pred))

            # Save model
            output = io.BytesIO()
            joblib.dump(model, output)
            st.download_button(
                label="💾 Download Trained Model",
                data=output.getvalue(),
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )

else:
    st.info("⬆️ Please upload a CSV file to get started.")
