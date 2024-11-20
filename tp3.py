import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import time
import seaborn as sns


## EXERCICE 1:

print("\n --- EXERCICE 1 --- \n")

### QUESTION 1:

print("\n --- QUESTION 1 --- \n")

# Charger le fichier CSV
data = pd.read_csv("spam.csv", encoding="latin-1")

# Nettoyer et renommer les colonnes
data = data.rename(columns={"v1": "label", "v2": "message"})[['label', 'message']]

# Convertir les labels en numérique
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Aperçu des données
print(data.head())
print(data.info())

# Distribution des classes
print("Distribution des classes :")
print(data['label'].value_counts(normalize=True))

print("Nombre total de messages :", len(data))


### QUESTION 2:

print("\n --- QUESTION 2 --- \n")

# Vérifier les valeurs manquantes
print("Valeurs manquantes : \n", data.isnull().sum())

# Retirer les espaces en début/fin des messages (si nécessaire)
data['message'] = data['message'].str.strip()

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

# Appliquer la vectorisation sur la colonne 'message'
X = vectorizer.fit_transform(data['message'])

# Label (target)
y = data['label']

# Afficher la forme des données vectorisées
print("Dimensions de la matrice vectorisée :", X.shape)


### QUESTION 3:

print("\n --- QUESTION 3 --- \n")

# Diviser les données en ensembles d'entraînement (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Afficher les tailles des ensembles
print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} messages")
print(f"Taille de l'ensemble de test : {X_test.shape[0]} messages")


### QUESTION 4:

print("\n --- QUESTION 4 --- \n")

# Initialiser le modèle Naïve Bayes Multinomial
nb_model = MultinomialNB()

# Entraîner le modèle sur l'ensemble d'entraînement
nb_model.fit(X_train, y_train)

# Afficher les paramètres appris par le modèle
print("Modèle Naïve Bayes entraîné avec succès.")


### QUESTION 5:

print("\n --- QUESTION 5 --- \n")

# Prédictions sur l'ensemble de test
y_pred_nb = nb_model.predict(X_test)

# Rapport de classification
print("Rapport de classification - Naïve Bayes :")
print(classification_report(y_test, y_pred_nb, target_names=["Ham", "Spam"]))

# Calcul et affichage de la matrice de confusion
cm_nb = confusion_matrix(y_test, y_pred_nb)
ConfusionMatrixDisplay(cm_nb, display_labels=["Ham", "Spam"]).plot(cmap="Blues")
plt.title("Matrice de Confusion - Naïve Bayes")
plt.show()


### QUESTION 6:

print("\n --- QUESTION 6 --- \n")

'''
Un modèle génératif comme Naïve Bayes est rapide et efficace pour des 
ensembles de données simples. Il suppose une indépendance conditionnelle 
entre les variables, ce qui le rend plus rapide mais parfois moins précis 
si cette hypothèse est invalide.

'''



## EXERCICE 2:

print("\n --- EXERCICE 2 --- \n")

### QUESTION 1:

print("\n --- QUESTION 1 --- \n")

# Charger le fichier CSV
data = pd.read_csv("spam.csv", encoding="latin-1")

# Nettoyer et renommer les colonnes
data = data.rename(columns={"v1": "label", "v2": "message"})[['label', 'message']]

# Convertir les labels en numérique
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Aperçu des données
print(data.head())
print(data.info())

# Distribution des classes
print("Distribution des classes :")
print(data['label'].value_counts(normalize=True))

print("Nombre total de messages :", len(data))


### QUESTION 2:

print("\n --- QUESTION 2 --- \n")

# Vérifier les valeurs manquantes
print("Valeurs manquantes : \n", data.isnull().sum())

# Retirer les espaces en début/fin des messages (si nécessaire)
data['message'] = data['message'].str.strip()

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

# Appliquer la vectorisation sur la colonne 'message'
X = vectorizer.fit_transform(data['message'])

# Label (target)
y = data['label']

# Afficher la forme des données vectorisées
print("Dimensions de la matrice vectorisée :", X.shape)


### QUESTION 3:

print("\n --- QUESTION 3 --- \n")

# Initialiser le modèle Complement Naive Bayes
cnb_model = ComplementNB()

# Entraîner le modèle sur l'ensemble d'entraînement
cnb_model.fit(X_train, y_train)

# Confirmation
print("Modèle Complement Naive Bayes entraîné avec succès.")


### QUESTION 4:

print("\n --- QUESTION 4 --- \n")

# Prédictions sur l'ensemble de test
y_pred_cnb = cnb_model.predict(X_test)

# Rapport de classification
print("Rapport de classification - Complement Naive Bayes :")
print(classification_report(y_test, y_pred_cnb, target_names=["Ham", "Spam"]))


### QUESTION 5:

print("\n --- QUESTION 5 --- \n")

# Calculer la matrice de confusion
cm_cnb = confusion_matrix(y_test, y_pred_cnb)

# Afficher la matrice de confusion
ConfusionMatrixDisplay(cm_cnb, display_labels=["Ham", "Spam"]).plot(cmap="Greens")
plt.title("Matrice de Confusion - Complement Naive Bayes")
plt.show()


### QUESTION 6:

print("\n --- QUESTION 6 --- \n")

'''
La matrice de confusion montre :

TP : Spams correctement détectés.
FP : Ham classés à tort comme Spam.
FN : Spams non détectés.
TN : Ham correctement détectés.

'''



## EXERCICE 3:

print("\n --- EXERCICE 3 --- \n")

### QUESTION 1:

print("\n --- QUESTION 1 --- \n")

# Charger le fichier CSV
data = pd.read_csv("spam.csv", encoding="latin-1")

# Nettoyer et renommer les colonnes
data = data.rename(columns={"v1": "label", "v2": "message"})[['label', 'message']]

# Convertir les labels en numérique
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Aperçu des données
print(data.head())
print(data.info())

# Distribution des classes
print("Distribution des classes :")
print(data['label'].value_counts(normalize=True))

print("Nombre total de messages :", len(data))


### QUESTION 2:

print("\n --- QUESTION 2 --- \n")

# Vérifier les valeurs manquantes
print("Valeurs manquantes : \n", data.isnull().sum())

# Retirer les espaces en début/fin des messages (si nécessaire)
data['message'] = data['message'].str.strip()

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

# Appliquer la vectorisation sur la colonne 'message'
X = vectorizer.fit_transform(data['message'])

# Label (target)
y = data['label']

# Afficher la forme des données vectorisées
print("Dimensions de la matrice vectorisée :", X.shape)


### QUESTION 3:

print("\n --- QUESTION 3 --- \n")

# Diviser les données en ensembles d'entraînement (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Afficher les tailles des ensembles
print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} messages")
print(f"Taille de l'ensemble de test : {X_test.shape[0]} messages")


### QUESTION 4:

print("\n --- QUESTION 4 --- \n")

# Initialiser le modèle de régression logistique
lr_model = LogisticRegression(max_iter=1000)

# Entraîner le modèle sur l'ensemble d'entraînement
lr_model.fit(X_train, y_train)

# Confirmation
print("Modèle Régression Logistique entraîné avec succès.")


### QUESTION 5:

print("\n --- QUESTION 5 --- \n")

# Prédictions sur l'ensemble de test
y_pred_lr = lr_model.predict(X_test)

# Rapport de classification
print("Rapport de classification - Régression Logistique :")
print(classification_report(y_test, y_pred_lr, target_names=["Ham", "Spam"]))

# Calcul de la matrice de confusion
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Affichage de la matrice de confusion
ConfusionMatrixDisplay(cm_lr, display_labels=["Ham", "Spam"]).plot(cmap="Purples")
plt.title("Matrice de Confusion - Régression Logistique")
plt.show()


### QUESTION 6:

print("\n --- QUESTION 6 --- \n")


'''
Un modèle discriminant comme la régression logistique est plus flexible 
et performant sur des données complexes ou bruitées. Il n'exige pas 
d'hypothèse forte comme celle d'indépendance.

'''



## EXERCICE 4:

print("\n --- EXERCICE 4 --- \n")

### QUESTION 1:

print("\n --- QUESTION 1 --- \n")

# Calcul des métriques pour chaque modèle
results = {
    "Modèle": ["Naïve Bayes", "Complement Naïve Bayes", "Régression Logistique"],
    "Précision": [
        precision_score(y_test, y_pred_nb),
        precision_score(y_test, y_pred_cnb),
        precision_score(y_test, y_pred_lr),
    ],
    "Rappel": [
        recall_score(y_test, y_pred_nb),
        recall_score(y_test, y_pred_cnb),
        recall_score(y_test, y_pred_lr),
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_nb),
        f1_score(y_test, y_pred_cnb),
        f1_score(y_test, y_pred_lr),
    ],
}

# Créer un DataFrame pour afficher les résultats
results_df = pd.DataFrame(results)
print(results_df)


### QUESTION 2:

print("\n --- QUESTION 2 --- \n")

# Réduction de la dimensionnalité à 2D avec PCA
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train.toarray())
X_test_2D = pca.transform(X_test.toarray())

# Remplacer les valeurs négatives par 0 après PCA
X_train_2D[X_train_2D < 0] = 0
X_test_2D[X_test_2D < 0] = 0

# Entraîner les modèles sur les données réduites
nb_model.fit(X_train_2D, y_train)
cnb_model.fit(X_train_2D, y_train)
lr_model.fit(X_train_2D, y_train)

# Fonction pour afficher les frontières de décision
def plot_decision_boundary(X, y, model, title):
    h = .02  # Taille de pas pour la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=30)
    plt.title(title)

# Tracer les frontières de décision pour chaque modèle
plt.figure(figsize=(12, 4))

# Naïve Bayes
plt.subplot(131)
plot_decision_boundary(X_train_2D, y_train, nb_model, "Naïve Bayes")

# Complement Naïve Bayes
plt.subplot(132)
plot_decision_boundary(X_train_2D, y_train, cnb_model, "Complement Naïve Bayes")

# Régression Logistique
plt.subplot(133)
plot_decision_boundary(X_train_2D, y_train, lr_model, "Régression Logistique")

plt.show()


### QUESTION 3:

print("\n --- QUESTION 3 --- \n")

# Fonction pour mesurer le temps d'entraînement
def train_and_measure_time(model, X_train, y_train):
    start_time = time.time()  # Enregistrer le temps avant l'entraînement
    model.fit(X_train, y_train)  # Entraîner le modèle
    end_time = time.time()  # Enregistrer le temps après l'entraînement
    return end_time - start_time  # Retourner le temps écoulé

# Mesurer le temps d'entraînement pour chaque modèle
time_nb = train_and_measure_time(nb_model, X_train_2D, y_train)
time_cnb = train_and_measure_time(cnb_model, X_train_2D, y_train)
time_lr = train_and_measure_time(lr_model, X_train_2D, y_train)

# Affichage des temps d'entraînement pour chaque modèle
print(f"Temps d'entraînement - Naïve Bayes : {time_nb:.4f} secondes")
print(f"Temps d'entraînement - Complement Naïve Bayes : {time_cnb:.4f} secondes")
print(f"Temps d'entraînement - Régression Logistique : {time_lr:.4f} secondes")


### QUESTION 4:

print("\n --- QUESTION 4 --- \n")

# Calcul des F1-scores pour chaque modèle
f1_scores = [
    f1_score(y_test, y_pred_nb),
    f1_score(y_test, y_pred_cnb),
    f1_score(y_test, y_pred_lr)
]

# Création d'un DataFrame pour faciliter la visualisation
f1_data = pd.DataFrame({
    'Modèle': ['Naïve Bayes', 'Complement Naïve Bayes', 'Régression Logistique'],
    'F1-Score': f1_scores
})

# Visualisation des F1-scores avec un graphique en barres
plt.figure(figsize=(8, 6))
sns.barplot(x='Modèle', y='F1-Score', data=f1_data, palette='viridis')
plt.title('Comparaison des F1-Scores des Modèles')
plt.xlabel('Modèles')
plt.ylabel('F1-Score')
plt.show()
