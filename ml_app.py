from termios import TAB2
import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


st.title('Machine Learning App')
st.write("""
#### Exploring different ML models for Classification
Which one performs the best ?
""")
st.write("")
tab1, tab2 = st.tabs(["Home","Dataset"])

with tab1:

    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer","Wine"))
    classifier_name = st.sidebar.selectbox("Select ML Classifier", 
        ("K-Nearest Neighbors", 
        "Support Vector Machine",
        "Random Forest",
        "XGBoost"))

    # Retrieve Dataset
    def get_dataset(dataset_name):
        if dataset_name == "Iris":
            data = datasets.load_iris()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        elif dataset_name == "Wine":
            data = datasets.load_wine()
        X = data.data
        y = data.target

        return X, y

    X, y = get_dataset(dataset_name)

    st.write("##### Summary")

    with st.container():
        st.write("Number of dataset:", X.shape[0])
        st.write("Number of features:", X.shape[1])
        st.write("Number of target classes:", len(np.unique(y)))

    # Define model parameters
    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == "K-Nearest Neighbors":
            K = st.sidebar.slider("K",1,20)
            params["K"] = K
        elif clf_name == "Support Vector Machine":
            c = st.sidebar.slider("c",0.01,10.0)
            kernel = st.sidebar.radio("Kernel Function", ("rbf","linear","poly","sigmoid"))
            params["c"] = c
            params["kernel"] = kernel
        elif clf_name == "Random Forest":
            max_depth = st.sidebar.slider("max_depth",2,15)
            n_estimators = st.sidebar.slider("n_estimators",10,150)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        elif clf_name == "XGBoost":
            eta = st.sidebar.slider("Learning Rate",0.01,1.0)
            n_estimators = st.sidebar.slider("n_estimators",10,150)
            max_depth = st.sidebar.slider("max_depth", 2,15)
            reg_lambda = st.sidebar.slider("lambda", 0.1,5.0)
            params["eta"] = eta
            params["n_estimators"] = n_estimators
            params["max_depth"] = max_depth
            params["lambda"] = reg_lambda
        return params

    params = add_parameter_ui(classifier_name)

    # Define Classifiers
    def get_classifier(clf_name, params):
        if clf_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier(n_neighbors=params["K"])
        elif clf_name == "Support Vector Machine":
            model = SVC(C=params["c"], kernel=params["kernel"])
        elif clf_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"], random_state=42)
        elif clf_name == "XGBoost":
            model = XGBClassifier(eta=params["eta"], n_estimators=params["n_estimators"], max_depth=params["max_depth"], reg_lambda=params["lambda"])
        return model
        
    # Model Building & Evaluation
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=10)
    model = get_classifier(classifier_name, params)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test,y_pred)
    f1Score = f1_score(y_test, y_pred, average="weighted")


    st.write(f"Classifier: **{classifier_name}**")
    st.write(f"Model Accuracy: **{round(score*100,2)}%**")
    st.write(f"Weighted F1 Score:  **{round(f1Score*100,2)}%**")

    # Plotting Result
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(X)
    x1 = x_pca[:,0]
    x2 = x_pca[:,1]

    st.markdown("---")

    st.write("##### Data Visualization")
    st.write("Classification Plot")
    fig = plt.figure()
    plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    plt.title("Classification with 2 Major Principal Components")
    plt.colorbar()
    st.pyplot(fig)

    st.write("")
    st.write("")
    st.write("")
    st.write("Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g',ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Result')
    ax.set_ylabel('Actual Result')
    st.pyplot(fig)

with tab2:
    st.write(f"{dataset_name} Dataset")
    st.dataframe(X)

    st.download_button(
    label="Download data as CSV",
    data=pd.DataFrame(X).to_csv(index=False).encode('utf-8'),
    file_name=f"{dataset_name}.csv",
    mime='text/csv',
    )