#Bibliotheques
#Bibliotheques
import mysql.connector
import sqlite3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import string #
from dotenv import load_dotenv
# Charge les variables d'environnement depuis le fichier .env
load_dotenv()

from nltk.stem import WordNetLemmatizer

# tkenisation des mot en francais
import spacy

""" améliorer l'expérience de développement en permettant 
une autocomplétion plus agressive dans l'interpréteur IPython 
%config IPCompleter.greedy = True
%matplotlib inline """


""" améliorer l'expérience de développement en permettant 
une autocomplétion plus agressive dans l'interpréteur IPython 
%config IPCompleter.greedy = True
%matplotlib inline """

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#data

''' Step 1: ID code '''

#path

def connect_and_fetch_data(table_name, typedictionnaire=None):

    """ 
    Cree la connexion entre La base donnees et le code python.

    paramettre : nom des tables et 
                    le typdictionnaire ce trouvant dans la tables dictionnaire

    retourne: une table structurer et ces colonnes.

    """
    host = "localhost"
    database = "plateforme"
    user = "root"
    # password = " "

    # Établir une connexion à la base de données
    conn = mysql.connector.connect(host=host, database=database, user=user)

    try:
        # Créer un curseur pour exécuter des requêtes SQL
        cursor = conn.cursor()

        # Exécuter une requête SELECT pour récupérer toutes les données de la table spécifiée
        if table_name == "tbldictionnaire" and typedictionnaire is not None:
            query = f"SELECT * FROM tbldictionnaire WHERE typedictionnaire = '{typedictionnaire}'"
        else:
            query = f"SELECT * FROM {table_name}"
        cursor.execute(query)

        # Récupérer les résultats et les descriptions de colonnes
        results = cursor.fetchall()
        columns = [i[0] for i in cursor.description]

    finally:
        # Fermer le curseur et la connexion, même en cas d'erreur
        cursor.close()
        conn.close()

    return results,columns


def find_best_ridge_params(data, target):
  """
  Performs grid search with Ridge regression to find the best hyperparameters
  based on R-squared score.

  Args:
      data (pandas.Series): The text data to be vectorized.
      target (pandas.Series): The target variable (e.g., average salary).

  Returns:
      tuple: A tuple containing two dictionaries.
          - best_params: The dictionary containing the best hyperparameters found.
          - best_scores: The dictionary containing R-squared scores for training and test data.
  """

  max_dfs = [0.5, 0.7, 0.9]
  min_dfs = [1, 2, 5]
  ngram_ranges = [(1, 1), (1, 2), (2, 2)]
  alphas = [0.001, 0.01, 0.1, 1]

  best_score = -float('inf')  # Initialize with negative infinity
  best_params = {}

  for max_df in max_dfs:
    for min_df in min_dfs:
      for ngram_range in ngram_ranges:
        for alpha in alphas:
          tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
          X = tfidf.fit_transform(data)
          X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

          model = Ridge(alpha=alpha)
          model.fit(X_train, y_train)
          y_pred_train = model.predict(X_train)
          y_pred_test = model.predict(X_test)

          r2_score_train = r2_score(y_train, y_pred_train)
          r2_score_test = r2_score(y_test, y_pred_test)

          # Update best parameters and score if current score is better
          if r2_score_test > best_score:
            best_score = r2_score_test
            best_params = {
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": ngram_range,
                "alpha": alpha,
            }

  return best_params, {"train": best_score, "test": r2_score_train}

# # configure your API Key

nltk.download('stopwords')

# Créer deux ensembles de stop words, un pour le français et un pour l'anglais
stop_words_fr = set(nltk.corpus.stopwords.words('french'))
stop_words_en = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    """
    Netoye le texte, suprime les caractere speciaux, les caracteres html 
                et supprime les stop word rn francais et anglais

    parametre : le text

    retourne : un test propre

    """

    text = text.lower()
    text = re.sub(r'<[^>]*>|style=\"[^"]*\"', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])

    # Supprimer les stop words français et anglais
    all_stop_words = stop_words_fr.union(stop_words_en)  # Combiner les deux ensembles
    text = ' '.join([word for word in text.split() if word not in all_stop_words])

    return text
 
 
# Appliquer la fonction de nettoyage
# data['Clean_Description'] = data['Description'].apply(clean_text)
 
# Fonction de lemmatisation
def lemmatize_text(text, language='english'):
  """
  Lemmatizes text based on the specified language.

  Args:
      text: The text to lemmatize.
      language: The language of the text ('english' or 'french'). Defaults to 'english'.

  Returns:
      The lemmatized text as a string.
  """
  # Define lemmatizers for both languages
  lemmatizer_en = WordNetLemmatizer()
  lemmatizer_fr = WordNetLemmatizer()

  # Tokenize the text based on language
  if language == 'english':
    tokens = word_tokenize(text)
  elif language == 'french':
    # NLTK word_tokenize might not be ideal for French, consider spaCy
    tokens = [token.text.lower() for token in spacy.tokenizer(text)]  # Example using spaCy for French
  else:
    raise ValueError("Unsupported language: {}".format(language))

  # Lemmatize tokens based on language
  if language == 'english':
    lemmatized_tokens = [lemmatizer_en.lemmatize(token) for token in tokens]
  elif language == 'french':
    lemmatized_tokens = [lemmatizer_fr.lemmatize(token) for token in tokens]

  return ' '.join(lemmatized_tokens)


def calculer_cout(code_solution, dataframe):
    """
    Calcule le coût d'une solution. remplace les coûts inexistant par zeros

    Args:
        code_solution (str): Le code de la solution.
        dataframe (pandas.DataFrame): Le DataFrame contenant les informations sur les solutions.

    Returns:
        float: Le coût de la solution.
    """
    # Convertir les colonnes 'reelcoutrex', 'maxicoutrex' et 'minicoutrex' en double
    for col in ['reelcoutrex', 'maxicoutrex', 'minicoutrex']:
        dataframe[col] = dataframe[col].astype(float)
        
    # Remplacer les valeurs manquantes par la moyenne de chaque colonne
    for col in ['reelcoutrex', 'maxicoutrex', 'minicoutrex']:
        dataframe[col].fillna(0, inplace=True)
        
    # Filtrer le dataframe pour obtenir les lignes correspondant au code de solution
    lignes_code_solution = dataframe[dataframe['codesolution'] == code_solution]

    # Si le coût réel est disponible, le retourner
    if not lignes_code_solution.empty:
        cout_estime = lignes_code_solution['reelcoutrex'].iloc[0] if not pd.isnull(lignes_code_solution['reelcoutrex'].iloc[0]) else max(lignes_code_solution['maxicoutrex'].iloc[0], lignes_code_solution['minicoutrex'].iloc[0])
        return cout_estime

    # Sinon, retourner un message d'erreur
    else:
        return "Code de solution non trouvé dans le dataframe"

from sklearn.model_selection import learning_curve

# Définir une fonction pour créer la courbe d'apprentissage
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    elle affiche la courbe d'apprentissage du model Ridge

    paramettre : model, X_tarin, y_train

    Retourne : Une courbe contenant, les donnes entrainner et la crossvalidation

    """


    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training data")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
