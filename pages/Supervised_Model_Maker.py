import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor , VotingClassifier
from sklearn.svm import SVC , SVR
from sklearn.metrics import accuracy_score , r2_score , mean_squared_error , classification_report
from sklearn.neighbors import KNeighborsClassifier , KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
import io

st.set_page_config(page_title="Supervised Model Trainer", layout="wide")
st.title("Supervised Model Trainer")

uploaded_file = st.file_uploader("Upload Your CSV file",type=["csv"])

if uploaded_file :
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())


    target = st.multiselect("Select the target column (select exactly one):", df.columns, max_selections=1)
    if target:
        target = target[0] 
        features = st.multiselect("Select feature columns:", [col for col in df.columns if col != target])

        if features and target :
            X = df[features]
            y = df[target]

            # Convert target to string if it's object dtype for clean comparison
            y_unique = y.dropna().unique()

            # Normalize to string and lower-case for robustness
            y_str_values = set(str(val).strip().lower() for val in y_unique)

            # Check binary classification by string or numeric values
            if len(y_str_values) <= 20 and (
                pd.api.types.is_integer_dtype(y)
                or pd.api.types.is_bool_dtype(y)
                or y_str_values.issubset({"0", "1", "true", "false", "yes", "no"})
            ):
                task_type = "classification"
            else:
                task_type = "regression"

            st.write(f"Detected task type as {task_type}")

        test_size = st.slider("Slect Test Size(for splitting):",0.1,0.5,0.2)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)

        classification_model = [
            "Logistic Regression" ,
            "Random Forest Classifier" ,
            "Support Vector Classifier" ,
            "k-Nearest Neighbors" ,
            "Decision Tree",
            "Voting Classifier (Combine Models)"
        ]

        regression_models = [
            "Linear Regression" , 
            "Random Forest Regressor" ,
            "Support Vector Regressor" ,
            "k-Nearest Neighbors Regressor" ,
            "Decision Tree Regressor"
        ]

        model_type = st.selectbox("Choose a Model", classification_model if task_type == "classification" else regression_models)

        model = None

        if model_type == "Logistic Regression":
            c_val = st.slider("C (Inverse of regularization strength)",0.01,10,1)
            max_iter = st.slider("Max Iterations",100,2000,1000)
            model = LogisticRegression(C=c_val,max_iter=max_iter)

        elif model_type == "Random Forest Classifier":
            n_estimators = st.slider("Number of Trees",10,100,50)
            max_depth = st.slider("Max Depth",1,10,5)
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)

        elif model_type == "Support Vector Classifier":
            C = st.slider("C (Regularization parameter)",0.01,10,1)
            kernel = st.selectbox("Kernel",["linear","poly","rbf","sigmoid"])
            model = SVC(C=C,kernel=kernel)

        elif model_type == "Linear Regression":
            model = LinearRegression()

        elif model_type == "Random Forest Regressor":
            n_estimators = st.slider("Number of Trees",10,100,50)
            max_depth = st.slider("Max Depth",1,10,5)
            model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)

        elif model_type == "Support Vector Regressor":
            C = st.slider("C (Regularization parameter)",0.01,10,1)
            kernel = st.selectbox("Kernel",["linear","poly","rbf","sigmoid"])
            model = SVR(C=C,kernel=kernel)
        
        elif model_type == "k-Nearest Neighbors":
            n_neighbors = st.slider("Number of Neighbors",1,20,5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        elif model_type == "k-Nearest Neighbors Regressor":
            n_neighbors = st.slider("Number of Neighbors",1,20,5)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)

        elif model_type == "Decision Tree":
            criterion = st.selectbox("Criterion",["gini","entropy"])
            max_depth = st.slider("Max Depth",1,50,5)
            model = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth) 

        elif model_type == "Decision Tree Regressor":
            criterion = st.selectbox("Criterion",["gini","entropy"])
            max_depth = st.slider("Max Depth",1,50,5)
            model = DecisionTreeRegressor(criterion=criterion,max_depth=max_depth)
        
        elif model_type == "Voting Classifier (Combine Models)":
            models = [
                "Logistic Regression" ,
                "Random Forest Classifier" ,
                "Support Vector Classifier" ,
                "k-Nearest Neighbors" ,
                "Decision Tree"
            ]
            model = VotingClassifier(estimators=[LogisticRegression() , RandomForestClassifier() , SVC() , KNeighborsClassifier() , DecisionTreeClassifier()])

        if st.button("Train Model"):
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            st.subheader("Model Performance")
            if task_type == "regression":
                mse = mean_squared_error(y_test,y_pred)
                r2 = r2_score(y_test,y_pred)
                st.write(f"Mean Squared Error : {mse}")
                st.write(f"R2 Score : {r2}")
            else:
                acc = accuracy_score(y_test,y_pred)
                st.write(f"Accuracy Score : {acc}")
                st.text("Classification Report:")
                st.text(classification_report(y_test,y_pred))

            output = io.BytesIO()
            joblib.dump(model,output)
            st.download_button(
                label="Download Trained Model",
                data=output.getvalue(),
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )

else:
    st.info("Please upload a CSV file")