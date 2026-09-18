import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


st.write(''' # Predicción de categoría de Premio Nobel ''')
st.image("Nobel.png", caption="El Titanic navegaba desde Southampton, Inglaterra, hasta Nueva York en Estados Unidos.")

st.header('Texto')

def user_input_features():
  # Entrada
  texto = st.text_input("Introduce el texto a evaluar")

  user_input_data = {'Text': texto}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

nobel =  pd.read_csv('df_nobel.csv', encoding='latin-1')
X = nobel.Text
y = nobel.Label

vect = CountVectorizer()
vect.fit(X)
X_dtm = vect.transform(X)
X_dtm = vect.fit_transform(X)

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_dtm)
tfidf_transformer.transform(X_dtm)

nb = MultinomialNB()
nb.fit(X_dtm, y)

df_dtm = vect.transform(df)
prediction = nb.predict(df_dtm)

#{'physics':0, 'medicine':1, 'peace':2, 'literature':3, 'chemistry':4, 'economics':5}
#'Physics', 'Medicine', 'Peace', 'Literature', 'Chemistry', 'Economics'
st.subheader('Predicción')
if prediction == 0:
  st.write('Physics')
elif prediction == 1:
  st.write('Medicine')
elif prediction == 2:
  st.write('Peace')
elif prediction == 3:
  st.write('Literature')
elif prediction == 4:
  st.write('Chemistry')
elif prediction == 5:
  st.write('Economics')
else:
  st.write('Sin predicción')
