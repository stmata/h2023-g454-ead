import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predictive Modelling
def ml_fucntion(X, Y):
    test_proportion = 0.30
    seed = 5
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_proportion,
                                                        random_state=seed)
    transform_options = ["None", 
                         "StandardScaler", 
                         "Normalizer", 
                         "MinMaxScaler"]
    transform = st.selectbox("Select data transform",
                             transform_options)
    if transform == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif transform == "Normalizer":
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif transform == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train
        X_test = X_test
    classifier_list = ["LogisticRegression",
                       "SVM",
                       "DecisionTree",
                       "KNeighbors",
                       "RandomForest"]
    classifier = st.selectbox("Select classifier", classifier_list)
    # Add option to select classifiers
    # Add LogisticRegression
    if classifier == "LogisticRegression":
        st.write("Here are the results of a logistic regression model:")
        solver_value = st.selectbox("Select solver",
                                    ["lbfgs",
                                     "liblinear",
                                     "newton-cg",
                                     "newton-cholesky"])
        model = LogisticRegression(solver=solver_value)
        model.fit(X_train, y_train)
        # Make prediction
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average = "weighted")
        # Display dresults
        st.write(f'Accuracy: {accuracy}')
        st.write(f'Precision: {precision}')
        st.write(f'Recall: {recall}')
        st.write(f'F1-Score: {f1}')
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
    elif classifier == "DecisionTree":
        st.write("Here are the results of a logistic regression model:")
        
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        # Make prediction
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average = "weighted")
        # Display dresults
        st.write(f'Accuracy: {accuracy}')
        st.write(f'Precision: {precision}')
        st.write(f'Recall: {recall}')
        st.write(f'F1-Score: {f1}')
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
    elif classifier == "RandomForest":
        st.write("Here are the results of a Random Forest model:")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Display results
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 score: {f1}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
    # Add SVM option
    elif classifier == "SVM":
        st.write("Here are the results of an SVM model:")
        regularization = st.selectbox("Select C value", [1.0, 1.5, 2.0, 2.5, 3.0])
        kernel_value = st.selectbox("Select kernel value", ['linear', 'poly', 'rbf', 'sigmoid'])
        model = SVC(C=regularization, kernel=kernel_value)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Display results
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 score: {f1}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
    # Add SVM option
    elif classifier == "KNeighbors":
        st.write("Here are the results of a K-Nearest Neighbors model:")
        k = st.slider("Select the value of k:", min_value=1, max_value=20, step=1)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 score: {f1}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
           

