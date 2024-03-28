import streamlit as st #API
from DatabaseCleanBestParameters import *
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import joblib
import time
# Mesure de l'utilisation du CPU
import psutil 
from sklearn.metrics.pairwise import cosine_similarity

# Début du chronométrage
start_time = time.time()
path_data_csv = os.getenv("path_db")

# # Charger le modèle

with open(r'C:\Users\Férol\Documents\BattleIAPau_2024\model_Ridge.pkl', 'rb') as file:
    model_Ridge = pickle.load(file)

def load_results():
    with open(r'C:\Users\Férol\Documents\BattleIAPau_2024\results_evaluation.pkl', 'rb') as file:
        results = pickle.load(file)
    return results

# Charger les résultats
results = load_results()

def load_results():
    with open(r'C:\Users\Férol\Documents\BattleIAPau_2024\result_data_train_test.pkl', 'rb') as file:
        results_data_train_test = pickle.load(file)
    return results_data_train_test

# Charger les résultats
results_data_train_test = load_results()

# Charger le vecteuriseur TF-IDF ajusté
loaded_tfidf = joblib.load(r'C:\Users\Férol\Documents\BattleIAPau_2024\vecteuriseur_tfidf_fit.pkl')

""" # L’IA AU SERVICE DE LA TRANSITION ÉNERGÉTIQUE DES ENTREPRISES"""
# Interface utilisateur Streamlit
# st.title('Affichage des données')
# # Interface utilisateur Streamlit
st.title("Affichage des données depuis la base de données")
table_name = st.text_input("Nom de la table : ")

if table_name:
    if table_name == "tbldictionnaire":
        typedictionnaire = st.text_input("Type de dictionnaire : ")
        results, columns = connect_and_fetch_data(table_name, typedictionnaire)
    else:
        results, columns = connect_and_fetch_data(table_name)

    # Afficher les résultats dans Streamlit sous forme de DataFrame
    if results:
        df_affiche = pd.DataFrame(results, columns=columns)
        st.write("Résultats de la requête :")
        st.write(df_affiche)
    else:
        st.write("Aucun résultat trouvé.")

#recharger les donnees la suite 
results_use, columns_use = connect_and_fetch_data("tbldictionnaire", "sol")
df = pd.DataFrame(results_use, columns=columns_use)
df['CodeIdentifications'] = pd.to_numeric(df["codeappelobjet"])

# Afficher la taille des données
st.write(f'Taille des données de la table Solution : {df.shape}')

# Graphiques
st.header("Graphiques")

# # Display the graph directly with title
# st.title('Graphique de Code Identifications')
# st.line_chart(df['CodeIdentifications'], use_container_width=True)
st.title('Graphique de Code Identifications')

# Créer le graphique avec matplotlib
plt.figure(figsize=(10, 5))
plt.plot(df['CodeIdentifications'])
plt.title('Graphique de Code Identifications')
plt.xlabel('Index')  # Nom de l'axe des x
plt.ylabel('Code Solution')  # Nom de l'axe des y
# Afficher le graphique avec Streamlit
st.pyplot(plt.gcf(), use_container_width=True)

"""
Ce graphiques nous montres que le code identifiant semble suivre une 
progression lineaire et que ces valeurs a predire ne sont pas unique donc il semblerait que
nous devrions utiliser un model de regression lineaire
"""

# Histogramme du code d'identification with title
st.title('les mots les plus frequents dans la description')
all_words = ' '.join(df['traductiondictionnaire'].apply(clean_text)).split()

# Calcul de la fréquence des mots
word_freq = nltk.FreqDist(all_words)
 
# Récupération des 20 mots les plus fréquents
top_words = word_freq.most_common(20)
words, frequencies = zip(*top_words)
colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
 
 
fig = plt.figure(figsize=(10,6))
for word, frequency, color in zip(words, frequencies, colors):
    plt.bar(word, frequency, color=color)
 
plt.title('Top 20 mots fréquents')
plt.xlabel('Mots')
plt.ylabel('Fréquence')
plt.xticks(rotation=45)
st.pyplot(fig)




# Affichage des métriques
st.header("Métriques du modèle")
# Fin du chronométrage
end_time = time.time()

# Calcul du temps d'exécution
execution_time = end_time - start_time

# MSE et R² sur les données d'entraînement et de test
st.write(f"Mean Squared Error (MSE) sur les données d'entraînement : {results['mse_train']}")
st.write(f"Mean Squared Error (MSE) sur les données de test : {results['mse_test']}")
st.write(f"R-squared (R²) sur les données d'entraînement : {results['r2_train']}")
st.write(f"R-squared (R²) sur les données de test : {results['r2_test']}")


# Affichage du temps d'exécution
st.write(f"Temps d'exécution : {execution_time:.2f} secondes")

"""

#### Donc $97$% 
de la variance de 
la variable dependante est expliqué par le modèle

le modèle semble bien performer sur l'ensemble d'entraînement 
(selon le R² élevé), mais il montre des signes de surajustement sur 
l'ensemble de test, comme indiqué par le MSE du test plus élevés sur cet ensemble. 
Il pourrait être nécessaire d'ajuster (avec plus de donnees) le modèle pour améliorer sa capacité de 
généralisation."""

plot_learning_curve(model_Ridge, "Courbe d'apprentissage (Ridge)", results_data_train_test['X_train'], results_data_train_test['y_train'], cv=5, n_jobs=-1)
st.pyplot(plt)

"""Ce graphique  de la courbe d'apprentissage pour la régression Ridge. semble nous indiqué les performances du modèle avec différents volumes de données d'entraînement. Voici quelques points clés à noter :

- Score d'entraînement : Il semble que le score d'entraînement diminue progressivement à mesure que la taille de l'ensemble d'entraînement augmente. Cela est typique des modèles d'apprentissage automatique, car avec plus de données, il devient plus difficile pour le modèle de s'adapter parfaitement à toutes les données d'entraînement.

- Score de validation croisée : Le score de validation croisée semble augmenter avec la taille de l'ensemble d'entraînement. C'est un bon signe, car cela signifie que le modèle devient meilleur pour généraliser à de nouvelles données.

- Convergence des scores : À un certain point, les deux scores semblent converger, ce qui indique que le modèle a atteint un équilibre entre le biais (erreur due à des hypothèses simplificatrices) et la variance (erreur due à la sensibilité aux petites fluctuations dans les données d'entraînement).
"""



def predict_code(description):
    # Prétraiter la description
    description = clean_text(description)
    description = lemmatize_text(description)
    description = [description]
    # Vectoriser la description
    X_new = loaded_tfidf.transform(description)
    # Prédire le code identifiant
    y_pred = model_Ridge.predict(X_new)
    return y_pred[0]

def comparer_et_rechercher(description, df):
  """
  Compare la description du test avec les descriptions de la base de données et
  retourne les 3 résultats les plus proches.

  Args:
    description: La description du test à comparer.
    df: Le DataFrame contenant les descriptions et les codes d'identification.

  Returns:
    Un dictionnaire contenant les 3 résultats les plus proches, avec leur code d'identification et la similarité cosinus.
  """

  # Prétraitement de la description du test
  description_test = clean_text(description)
  description_test = lemmatize_text(description_test)

  # Vecteur TF-IDF de la description du test
  vec_test = loaded_tfidf.transform([description_test])

  # Calcul de la similarité cosinus entre la description du test et toutes les descriptions de la base
  similarites = cosine_similarity(vec_test, loaded_tfidf.transform(df['traductiondictionnaire']))

  # Tri des résultats par similarité décroissante
  indices_triés = np.argsort(similarites, axis=1)[:, -3:]

  # Dictionnaire des résultats
  resultats = {}
  for i, indices in enumerate(indices_triés):
    resultats[f"Résultat {i+1}"] = {
      "code": df['CodeIdentifications'].iloc[indices[0]],
      "similarité": similarites[0][indices[0]]
    }

  return resultats

# Interface utilisateur Streamlit
st.title('Application de prédiction et coût')

#Charger un fichier csv
uploaded_file = st.file_uploader("Charger un fichier CSV", type=['csv'])

if uploaded_file is not None:
    #chager le fichier Csv
    df_test = pd.read_csv(uploaded_file, encoding='latin-1', delimiter=";")
    #Demander a l'utilisateur de choisir la colonne contenant les descriptions
    colonne_description = st.selectbox(
        "Choisissez la colonne contenant les descriptions", df_test.columns,
    )
else:
    df_test = pd.DataFrame()

# Zone de saisie pour l'utilisateur
user_input = st.text_input('Entrez une description :', '')
counter = 1
if user_input or uploaded_file is not None:
    if user_input:
        descriptions = [user_input]
    else:
        descriptions = df_test[colonne_description].tolist()
    for user_input in descriptions:
      # Prédiction du code identifiant
      prediction = predict_code(user_input)

      # Comparaison et recherche des résultats les plus proches
      resultats = comparer_et_rechercher(user_input, df)

      # Affichage des résultats
      st.write(f"Description: {user_input}")
      st.write(f"Code identifiant prédit: {prediction}")
      st.write(f"Résultats les plus proches Numeros {counter}:")
      counter += 1
      for i, resultat in resultats.items():
        # st.write(f"Resultat {counter}:")
        st.write(f"  - Code: {resultat['code']}")
        st.write(f"  - Similarité cosinus: {resultat['similarité']}")
        # Définir le code de la solution
        code_solution_donner = resultat['code']

        # Supposons que vous avez une fonction connect_and_fetch_data qui récupère les résultats des coûts
        # et crée un dataframe à partir de ces résultats
        results_cout, columns_cout = connect_and_fetch_data("tblcoutrex")
        dataframe_cout = pd.DataFrame(results_cout, columns=columns_cout)
        # Calculer le coût
        cout_calculer = calculer_cout(code_solution_donner, dataframe_cout)

        # Afficher le coût
        st.write("Le coût de la solution {} est de : {} ".format(code_solution_donner, cout_calculer))
        

# Titre
st.write("# Mesure de l'utilisation du CPU")

# Mesure de l'utilisation du CPU : 
cpu_usage = psutil.cpu_percent()
st.write("Utilisation du CPU :", cpu_usage)
# """signifie que 62.7% des ressources du processeur (CPU) sont actuellement utilisées"""
