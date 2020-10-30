import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

sns.set()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Détection de fraudes de CB")


@st.cache
def load_data(path):
    return pd.read_csv(path)


def model_preds(model_class, X_train, y_train, X_test):
    model = model_class
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def main():
    df_raw = load_data("../data/creditcard.csv")
    fraud_df = df_raw[df_raw['Class'] == 1]
    non_fraud_df = df_raw[df_raw['Class'] == 0]
    line_nb = df_raw.shape[0]

    """ 1. Prévisualisation des données """
    st.markdown('''## Prévisualiation''')

    st.write("DataFrame brut:")
    st.write(df_raw.sample(frac=0.1))
    st.markdown('''
    Explications des variables du dataset :
    * Time : Temps passé entre l'action de l'utilisateur et la première transaction du dataset
    * V1, V2, ... : Les principales features relevées suite à une PCA
    * Amount : Somme dépensée lors de la transaction
    * Class : Prend la valeur 1 en cas de fraude, 0 sinon
    ''')

    """ 2. Analyse du dataset (utilisation de diagrammes) """
    st.markdown('''## Analyse exploratoire''')

    st.markdown(f'''**Nombre total de lignes** : {line_nb}''')
    st.markdown(f'''**Nombre total de variables** : {df_raw.shape[1]}''')
    st.markdown(f'''**Nombre de fraudes** : {fraud_df.shape[0]} ({round(100 * fraud_df.shape[0] / line_nb, 2)}%)''')
    st.markdown('''**Nombres de transactions normales** : {non_fraud_df.shape[0]} 
    ({round(100 * non_fraud_df.shape[0] / line_nb, 2)}%)''')

    st.markdown("### Histogramme du nombre de transactions/fraudes en fonction du temps")
    fig = px.histogram(fraud_df, x='Time', color='Class', title="Fraudes", nbins=40)
    st.plotly_chart(fig)

    fig = px.histogram(non_fraud_df, x='Time', color='Class', title="Vraies transactions", nbins=40)
    st.plotly_chart(fig)

    st.markdown("### Histogramme du nombre de transactions/fraudes en fonction du montant")
    fig = px.histogram(fraud_df, x='Amount', color='Class', title="Fraudes")
    st.plotly_chart(fig)

    fig = px.histogram(non_fraud_df, x='Amount', color='Class', title="Vraies transactions", nbins=40)
    st.plotly_chart(fig)

    st.markdown("### Diagramme en boite des transactions/fraudes en fonction du montant")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))

    ax[0].boxplot(non_fraud_df['Amount'], widths=0.5, patch_artist=True)
    ax[0].set_ylim(-20, 400)
    ax[0].set_xticklabels(['Regular'], fontsize=12)
    ax[0].set_ylabel('Amount', fontsize=12)

    ax[1].boxplot(fraud_df['Amount'], widths=0.5, patch_artist=True)
    ax[1].set_ylim(-20, 400)
    ax[1].set_xticklabels(['Fraud'], fontsize=12)

    st.pyplot(fig)

    corr = df_raw.corr()
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                linewidths=.1, cmap="RdBu", ax=ax)
    st.pyplot(fig)

    """ 3. Préparation des données pour leur utilisation dans des algorithmes de Machine Learning """
    st.header("Preprocessing")
    df_prep = df_raw.copy()

    # Normalization
    scaler = StandardScaler()
    df_prep['Amount_std'] = scaler.fit_transform(df_prep["Amount"].values.reshape(-1, 1))
    df_prep['Time_std'] = scaler.fit_transform(df_prep["Time"].values.reshape(-1, 1))
    df_prep = df_prep.drop(["Amount", "Time"], axis=1)

    # Splitting 80/20 for train/test
    X = df_prep.drop('Class', axis=1)
    y = df_prep['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    st.write(df_prep.sample(frac=0.01))

    # Balancing the dataset
    rus = RandomUnderSampler()
    X_rus, y_rus = rus.fit_sample(X_train, y_train)

    df_rus = X_rus.copy()
    df_rus['Class'] = y_rus
    corr = df_rus.corr()
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                linewidths=.1, cmap="RdBu", ax=ax)
    st.pyplot(fig)

    """ 4. Prédictions des données en utilisant des modèles de Machine Learning """
    st.markdown('''## Prédictions''')

    st.markdown('''### Régression logistique''')
    model, y_pred = model_preds(LogisticRegression(), X_rus, y_rus, X_test)
    st.write(classification_report(y_test, y_pred))
    st.write("AUC: {:.2f}\n".format(roc_auc_score(y_test, y_pred)))

    # confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True, ax=ax)

    ax.set_title("Matrice de corrélation")
    ax.set_ylabel("Valeure réelle")
    ax.set_xlabel("Prédite")
    st.pyplot()

    st.subheader("Decision Tree")
    model, y_pred = model_preds(DecisionTreeClassifier(), X_rus, y_rus, X_test)
    y_pred = model.predict(X_test)
    st.write(classification_report(y_test, y_pred))
    st.write("AUC: {:.2f}\n".format(roc_auc_score(y_test, y_pred)))

    # confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True, ax=ax)

    ax.set_title("Matrice de corrélation")
    ax.set_ylabel("Valeure réelle")
    ax.set_xlabel("Prédite")
    st.pyplot()

    dot = export_graphviz(model, filled=True, rounded=True, feature_names=X.columns, class_names=['0', '1'])
    # graph = pydotplus.graph_from_dot_data(dot)
    st.graphviz_chart(dot)
    # graph = pydotplus.graph_from_dot_data(dot)


if __name__ == "__main__":
    main()
