"""
Modulo para carga de archivo CSV
"""
#import datetime, spacy
import re
import os
#import calendar
#from time import time
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import TweetTokenizer
from nltk import ngrams
import nltk
import string
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
#from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.pipeline import Pipeline
#from collections import Counter
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import dataframe_image as dfi

class explorer():
    
    def __init__(self, stopwords):
        self.df_atenciones = pd.read_csv('Database/atenciones_database.csv',delimiter=',')
        self.df_atenciones['fecha_inicio'] = pd.to_datetime(self.df_atenciones['fecha_inicio'])
        self.df_atenciones['dia_inicio'] = pd.to_datetime(self.df_atenciones['fecha_inicio'].dt.date)
        self.df_atenciones['mes_inicio'] = self.df_atenciones['dia_inicio'].dt.month
        self.df_atenciones['ano_inicio'] = self.df_atenciones['dia_inicio'].dt.year
        self.df_atenciones['hechos'] = self.df_atenciones['hechos'].astype(str)
        self.df_atenciones['solucion'] = self.df_atenciones['solucion'].astype(str)
        self.df_atenciones['year_month'] = self.df_atenciones['fecha_inicio'].dt.to_period('M')
        self.df_atenciones['observaciones'] = self.df_atenciones['observaciones'].astype(str)
        self.stopwords_es = stopwords
        

    def clean_text(self, tweet):
        """
        Excluye menciones, emails, URLs y simbolos
        Args:
            tweet (any): Tweet a limpiar

        Returns:
            str: texto limpio
        """
        # Convertir a minusculas
        tweet = tweet.lower()
        # Excluir Palabras identificadas que no generan Valor (Luego implementamos un arreglo o una BD auxiliar que las guarde)  + Una libreria de sinonimos. + Lematizacion (Requerir-requiere)
        tweet = re.sub(r'usuario',' ', tweet)
        tweet = re.sub(r'usuaria',' ', tweet)
        tweet = re.sub(r'pues',' ', tweet)
        tweet = re.sub(r'fecha',' ', tweet)
        tweet = re.sub(r'mismo',' ', tweet)
        tweet = re.sub(r'dice',' ', tweet)
        tweet = re.sub(r'indica',' ', tweet)
        tweet = re.sub(r'cuenta',' ', tweet)
        tweet = re.sub(r'hace',' ', tweet)
        tweet = re.sub(r'solo',' ', tweet)
        tweet = re.sub(r'nro',' ', tweet)
        tweet = re.sub(r'vez',' ', tweet)
        # Excluir menciones o emails
        tweet = re.sub(r'\w*@(\w+\.*\w+\.*\w+)',' ', tweet)
        # Excluir simbolos
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Excluir URLs 
        tweet = re.sub(r'(?:www\.|https?)[^\s]+', ' ', tweet, flags=re.MULTILINE) 
        # Borrar espacios
        tweet = tweet.strip()
        # Considerar solo valores alfa numericos
        text_alfa = re.compile("^(?![0-9]*$)[a-zA-Z0-9]+$") 
        # Eliminar stopwords y palabras con longitud <= 2
        tokens = tweet.split()
        text = [token for token in tokens if token not in self.stopwords_es and len(token)>2 and text_alfa.match(token)]
        return ' '.join(text)
    # Aplicar filtro a texto
    def filter(self):
        df_atenciones_filtro = self.df_atenciones.copy()
        df_atenciones_filtro['hechos_filtrados'] = df_atenciones_filtro['hechos'].apply(self.clean_text)
        return df_atenciones_filtro

    def generate_word_cloud(self):
        """
        Las nubes de palabras o word clouds permiten visualizar las
        palabras más frecuentes de un texto utilizando el tamaño para
        representar la frecuencia o importancia. En este caso, las
        palabras se extraen de las atenciones filtradas (cerca de 8.160).

        Para volverlo un poco más ligado a la imagen del municipio,se utiliza
        un fondo con la bandera de Medellín pero se puede adaptar
        a cualquier tipo de imagen.
        """
        df_atenciones_filtro = self.filter()
        texto_tweets = ' '.join(df_atenciones_filtro['hechos_filtrados'])

        mask = np.array(Image.open('static/mask.png'))
        wordcloud_py = WordCloud(background_color="#F0F0F0", mode="RGBA",
            max_words=10000, mask=mask).generate(texto_tweets)
        image_colors = ImageColorGenerator(mask)
        fig = plt.figure(dpi=10000)
        plt.figure(figsize=[10,5])
        plt.imshow(wordcloud_py.recolor(color_func=image_colors), interpolation="bilinear")
        plt.axis("off")
        fig.patch.set_facecolor('#F0F0F0')
        buffer = BytesIO()
        plt.tight_layout(pad=0.1)
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        html = 'data:image/png;base64,{}'.format(base64_image)
        return html

    def generate_trigramas(self):
        """
        Los Trigramas son combinaciones de dos palabras que
        pueden dar una mejor idea de los temas de conversación.
        En este caso, es importante conocer los bigramas que más se
        repiten y para ellos se aplican técnicas de tokenización que
        separan las palabras del texto y cada par se convierte en
        una fila. También se puede modificar el tamaño del ngram
        para formar unigramas, trigramas, etc. El bigrama es una 
        opción intermedia que permite tener algo más de contexto 
        pero tiene suficientes ocurrencias para que sea significativa
        la muestra (mientras mayor sea el ngram, menor el número de ocurrencias).
        """
        ngram = 3
        tokenizer = TweetTokenizer()
        df_atenciones_filtro = self.filter()
        df_atenciones_filtro['tokenize'] = df_atenciones_filtro['hechos_filtrados'] \
            .apply(tokenizer.tokenize)
        df_atenciones_filtro['ngram'] = df_atenciones_filtro['tokenize'].apply(lambda x: list(ngrams(x, ngram)))
        df_atenciones_exploded = df_atenciones_filtro \
            .explode('ngram')[['dia_inicio','ano_inicio','ngram']]
        df_atenciones_exploded_grouped = df_atenciones_exploded \
        .groupby(['ano_inicio', 'ngram'])  \
        .agg({'dia_inicio':'count'}) \
        .reset_index() \
        .sort_values(by=['ano_inicio', 'dia_inicio'], ascending=False) \
        .rename(columns={'dia_inicio':'count'})
        df_atenciones_top_year = df_atenciones_exploded_grouped \
            .drop_duplicates(subset=['count','ano_inicio']) \
            .groupby(['ano_inicio']) \
            .head(10)
        plt.show()
        fig = plt.figure(dpi=1200,figsize=[10,2])
        sns.catplot(x="count",y="ngram",
            col="ano_inicio",
            data=df_atenciones_top_year, kind="bar",
            height=8, aspect=.7)
        fig.patch.set_facecolor('#F0F0F0')
        plt.tight_layout(pad=0.1)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        html = 'data:image/png;base64,{}'.format(base64_image)
        return html

    def generate_comunas(self):
        """
        Genera una tabla con las comunas mas afectadas
        """
        df_comunas = self.df_atenciones.groupby('comuna').count().reset_index()
        df_comunas = df_comunas.sort_values(by='comuna', ascending=True)[['comuna','solucion']].head(50)
        dfi.export(df_comunas, 'temps/comunas.png', max_cols=2)
        with open("temps/comunas.png", 'rb') as comunas_image:
            comunas_b64_image = base64.b64encode(comunas_image.read()).decode('utf-8')
        image_html = 'data:image/png;base64,{}'.format(comunas_b64_image)
        os.remove("temps/comunas.png")
        return image_html
    
    def generate_municipios(self):
        """
        Genera una tabla con los municipios mas afectados
        """
        df_municipios = self.df_atenciones.groupby('municipio').count().reset_index()
        df_municipios = df_municipios.sort_values(by='municipio', ascending=True)[['municipio','solucion']].head(50)
        dfi.export(df_municipios, 'temps/municipios.png', dpi=200, max_cols=2)
        with open("temps/municipios.png", 'rb') as municipios_image:
            municipios_b64_image = base64.b64encode(municipios_image.read()).decode('utf-8')
        image_html = 'data:image/png;base64,{}'.format(municipios_b64_image)
        os.remove("temps/municipios.png")
        return image_html