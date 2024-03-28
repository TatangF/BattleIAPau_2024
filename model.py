from DatabaseCleanBestParameters import *
import pickle
import joblib

# from sklearn.metrics.pairwise import cosine_similarity

#charger les donnee
results, columns = connect_and_fetch_data("tbldictionnaire", "sol")
df = pd.DataFrame(results, columns=columns)
# print(df_2.head())
# # Chargement des données
# path_data_csv = os.getenv("path_data_csv")
# df = charger_donnees(path_data_csv)

# Préparation des données
df['CodeIdentifications'] = pd.to_numeric(df["codeappelobjet"])
# df['descriptions'] = df["traductiondictionnaire"]
df['Description_clean'] = df["traductiondictionnaire"].apply(clean_text)
df['Description_clean']=  df['Description_clean'].apply(lemmatize_text)

X = df['Description_clean']
y = df['CodeIdentifications']
tfidf = TfidfVectorizer(max_features=None, min_df = 1, max_df=0.5, ngram_range=(1,2))

X = tfidf.fit_transform(X)

# Sauvegarder le vecteuriseur TF-IDF ajusté
joblib.dump(tfidf, 'vecteuriseur_tfidf_fit.pkl')

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #80 Train et 20 test 

# Modèle de régression
model_Ridge = Ridge(alpha=0.1)
model_Ridge.fit(X_train, y_train)


# Sauvegarder le modèle
with open('model_Ridge.pkl', 'wb') as file:
    pickle.dump(model_Ridge, file)


# Calcul des métriques d'évaluation
mse_train = mean_squared_error(y_train, model_Ridge.predict(X_train))
mse_test = mean_squared_error(y_test, model_Ridge.predict(X_test))

r2_train = r2_score(y_train, model_Ridge.predict(X_train))
r2_test = r2_score(y_test, model_Ridge.predict(X_test))

# Sauvegarder les résultats dans un dictionnaire
results_evaluation = {
    'mse_train': mse_train,
    'mse_test': mse_test,
    'r2_train': r2_train,
    'r2_test': r2_test
}

#sauvegarde des resultat de x_train, y_train, x_test, y_test
result_data_train_test = {
    'X_train' : X_train,
    'X_test' : X_test,
    'y_train' : y_train,
    'y_test' : y_test
}

# Sauvegarder le dictionnaire dans un fichier pickle
with open('result_data_train_test.pkl', 'wb') as file:
    pickle.dump(result_data_train_test, file)


# Sauvegarder le dictionnaire dans un fichier pickle
with open('results_evaluation.pkl', 'wb') as file:
    pickle.dump(results_evaluation, file)