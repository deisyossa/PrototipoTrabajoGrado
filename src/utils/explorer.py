"""
Modulo para carga de archivo CSV
"""
import re
import os
from time import time
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from collections import Counter
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
        plt.xlabel('Frecuencia')
        plt.ylabel('Trigrama')
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

    def generate_subtemas(self):
        """
        Genera una tabla con los subtemas tratados
        """
        df_sum_tema = self.df_atenciones.groupby('sub_tema').count().reset_index()
        df_sum_tema = df_sum_tema.sort_values(by='sub_tema', ascending=True)[['sub_tema','solucion']].head(50)
        dfi.export(df_sum_tema, 'temps/subtemas.png', dpi=200, max_cols=2)
        with open("temps/subtemas.png", 'rb') as subtemas_image:
            subtemas_b64_image = base64.b64encode(subtemas_image.read()).decode('utf-8')
        image_html = 'data:image/png;base64,{}'.format(subtemas_b64_image)
        os.remove("temps/subtemas.png")
        return image_html

    def generate_temas(self):
        """
        Genera una tabla con los temas principales tratados
        """
        df_sum_tema = self.df_atenciones.groupby('tema').count().reset_index()
        df_sum_tema = df_sum_tema.sort_values(by='tema', ascending=True)[['tema','solucion']].head(50)
        dfi.export(df_sum_tema, 'temps/temas.png', dpi=200, max_cols=2)
        with open("temps/temas.png", 'rb') as temas_image:
            temas_b64_image = base64.b64encode(temas_image.read()).decode('utf-8')
        image_html = 'data:image/png;base64,{}'.format(temas_b64_image)
        os.remove("temps/temas.png")
        return image_html

    def generate_servicios(self):
        """
        Genera una tabla con los servicios atendidos
        """
        df_servicios = self.df_atenciones.groupby('servicio').count().reset_index()
        df_servicios = df_servicios.sort_values(by='servicio', ascending=True)[['servicio','solucion']].head(50)
        dfi.export(df_servicios, 'temps/servicios.png', dpi=400, max_cols=2)
        with open("temps/servicios.png", 'rb') as servicios_image:
            servicios_b64_image = base64.b64encode(servicios_image.read()).decode('utf-8')
        image_html = 'data:image/png;base64,{}'.format(servicios_b64_image)
        os.remove("temps/servicios.png")
        return image_html

    def generate_usuarios(self):
        """
        Genera los graficos y tablas de los La cantidad de
        usuarios únicos que publican realizan solicitudes
        puede ayudar a identificar situaciones que provocaron
        mayores picos de participación en el tiempo. 
        """
        tokenizer = TweetTokenizer()
        ngram = 3
        buffer = BytesIO()
        df_atenciones_filtro = self.filter()
        df_atenciones_filtro['tokenize'] = df_atenciones_filtro['hechos_filtrados'] \
            .apply(tokenizer.tokenize)
        df_atenciones_filtro['ngram'] = df_atenciones_filtro['tokenize'].apply(lambda x: list(ngrams(x, ngram)))
        df_unique_users = df_atenciones_filtro.groupby(['dia_inicio'])['sub_tema'].nunique()
        df_unique_users = df_unique_users.reset_index().sort_values('sub_tema', ascending=False).head(5)
        dfi.export(df_unique_users, 'temps/tabla.png', dpi=200, max_cols=2)
        with open("temps/tabla.png", 'rb') as servicios_image:
            servicios_b64_image = base64.b64encode(servicios_image.read()).decode('utf-8')
        tabla_html = 'data:image/png;base64,{}'.format(servicios_b64_image)
        os.remove("temps/tabla.png")
        plt.show()
        fig = plt.figure(figsize=(15,5))
        sns.lineplot(x='dia_inicio', y='sub_tema', data=df_unique_users.reset_index())
        plt.title('Sub temas no repetidos por dia')
        plt.ylabel('Sub Tema')
        plt.xlabel('Fecha')
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        grafico_html = 'data:image/png;base64,{}'.format(base64_image)
        df_atenciones_exploded = df_atenciones_filtro \
            .explode('ngram')[['dia_inicio','ano_inicio','ngram']]
        plt.close('all')
        bar_html = self.generate_bar_usuario(df_atenciones_exploded)
        return grafico_html, tabla_html, bar_html    

    def generate_bar_usuario(self, df_atenciones_exploded):
        """
        Genera el grafico de barras
        """
        df_top_unique_days = df_atenciones_exploded[df_atenciones_exploded['dia_inicio'].isin(['2020-10-14','2020-10-15'])] \
            .groupby('ngram') \
            .count() \
            .reset_index() \
            .sort_values('dia_inicio', ascending=False) \
            .head(10)
        plt.show()
        fig = plt.figure(figsize=(20,5))
        sns.barplot(x='dia_inicio', y='ngram', data=df_top_unique_days, orient='h')
        plt.title('Top Trigramas para 14 y 15/11/2020')
        plt.xlabel('Menciones')
        plt.ylabel('Trigrama')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        bar_html = 'data:image/png;base64,{}'.format(base64_image)
        return bar_html

    @staticmethod
    def get_df_from_criteria(df, criteria):
        df = df[df['hechos_filtrados'] \
            .str \
            .contains(criteria, flags=re.IGNORECASE, regex=True)] \
            .groupby(['dia_inicio','ano_inicio','mes_inicio'], as_index=False) \
            .agg(['count'])['fecha_inicio'].reset_index().rename(columns={'count':'hechos'})
        return df.copy()

    def generate_eventos(self):
        df_atenciones_filtro = self.filter()

        filtro_incapacidad = r'\bincapacidad|\blicencia|\binasistencia'
        incapacidad_x_dia_df = self.get_df_from_criteria(df_atenciones_filtro, filtro_incapacidad)

        filtro_eps = r'\beps|\bentidad|\bprestadora|\bsalud'
        eps_x_dia_df = self.get_df_from_criteria(df_atenciones_filtro, filtro_eps)

        filtro_savia = r'\bsavia'
        savia_x_dia_df = self.get_df_from_criteria(df_atenciones_filtro, filtro_savia)

        filtro_sura = r'\bsura'
        sura_x_dia_df = self.get_df_from_criteria(df_atenciones_filtro, filtro_sura)   

        dia_html = self.get_grafico_dia(incapacidad_x_dia_df,
            eps_x_dia_df,savia_x_dia_df,sura_x_dia_df)
        sura_html = self.get_grafico_sura(sura_x_dia_df)
        eps_html = self.get_grafico_eps(eps_x_dia_df)
        savia_html = self.get_grafico_savia(savia_x_dia_df)
        return dia_html, sura_html, eps_html, savia_html
        
    @staticmethod
    def get_grafico_dia(incapacidad_x_dia_df,eps_x_dia_df,
            savia_x_dia_df,sura_x_dia_df):
        plt.show()
        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot(111)
        sns.lineplot(x='dia_inicio', y='hechos', data=incapacidad_x_dia_df[incapacidad_x_dia_df['mes_inicio'].isin([9,10,11])], ax=ax)
        sns.lineplot(x='dia_inicio', y='hechos', data=eps_x_dia_df[eps_x_dia_df['mes_inicio'].isin([9,10,11])], ax=ax)
        sns.lineplot(x='dia_inicio', y='hechos', data=sura_x_dia_df[sura_x_dia_df['mes_inicio'].isin([9,10,11])], ax=ax)
        sns.lineplot(x='dia_inicio', y='hechos', data=savia_x_dia_df[savia_x_dia_df['mes_inicio'].isin([9,10,11])], ax=ax)
        plt.title('Cantidad de Menciones por Dia')
        plt.ylabel('Menciones')
        plt.xlabel('Fecha')
        ax.legend(['incapacidad', 'eps', 'savia', 'sura'], loc='upper left', prop={'size': 12})
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        dia_html = 'data:image/png;base64,{}'.format(base64_image)
        return dia_html

    @staticmethod
    def get_grafico_sura(sura_x_dia_df):
        plt.show()
        fig = plt.figure(figsize=(15,6))
        sura_x_mes_df = sura_x_dia_df.groupby(['ano_inicio','mes_inicio']) \
                    .sum().reset_index()[['mes_inicio','ano_inicio','hechos']]
        sura_x_mes_pivot = sura_x_mes_df \
                            .pivot(index='mes_inicio',columns='ano_inicio', values='hechos')
        sura_x_mes_pivot.plot(kind='bar', figsize=(15, 6), color=['lightgray', 'gray', 'black'], rot=0)                                       
        plt.title("Comparaciones intermensuales de menciones de sura")
        plt.xlabel("Mes", labelpad=16)
        plt.ylabel("Menciones (sura)", labelpad=16)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        sura_html = 'data:image/png;base64,{}'.format(base64_image)
        return sura_html

    @staticmethod
    def get_grafico_eps(eps_x_dia_df):
        plt.show()
        fig = plt.figure(figsize=(15,6))
        eps_x_mes_df = eps_x_dia_df.groupby(['ano_inicio','mes_inicio']) \
                            .sum().reset_index()[['mes_inicio','ano_inicio','hechos']]
        eps_x_mes_pivot = eps_x_mes_df \
                            .pivot(index='mes_inicio',columns='ano_inicio', values='hechos')
        eps_x_mes_pivot.plot(kind='bar', figsize=(15, 6), color=['lightgray', 'gray', 'black'], rot=0)                                       
        plt.title("Comparaciones intermensuales de menciones de eps")
        plt.xlabel("Mes", labelpad=16)
        plt.ylabel("Menciones (eps)", labelpad=16)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        eps_html = 'data:image/png;base64,{}'.format(base64_image)
        return eps_html

    @staticmethod
    def get_grafico_savia(savia_x_dia_df):
        plt.show()
        fig = plt.figure(figsize=(15,6))
        savia_x_mes_df = savia_x_dia_df.groupby(['ano_inicio','mes_inicio']) \
                    .sum().reset_index()[['mes_inicio','ano_inicio','hechos']]
        savia_x_mes_pivot = savia_x_mes_df \
                            .pivot(index='mes_inicio',columns='ano_inicio', values='hechos')
        savia_x_mes_pivot.plot(kind='bar', figsize=(15, 6), color=['lightgray', 'gray', 'black'], rot=0)                                       
        plt.title("Comparaciones intermensuales de menciones de Savia EPS")
        plt.xlabel("Mes", labelpad=16)
        plt.ylabel("Menciones (Savia)", labelpad=16)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        savia_html = 'data:image/png;base64,{}'.format(base64_image)
        return savia_html

    def generate_topicos(self):
        """
        Existen diferentes técnicas de identificación de temas o tópicos pero una de las más utilizadas es
        Latent Dirichlet Allocation o LDA. Se trata de una técnica que genera un modelo probabilístico que
        asume que cada tema es una combinación de palabras y que cada documento (o caso en este escenario) es
        una combinación de temas con diferentes probabilidades.

        En las celdas de abajo, se crea un Pipeline que primero aplica una técnica conocida como TF-IDF que 
        calcula la frecuencia de palabras en los casos, y calcula un score para cada palabra dando menos 
        importancia a aquellas que aparecen con demasiada frecuencia y son poco relevantes. En el siguiente 
        paso del pipeline se entrena el modelo y transforma el dataframe original. En este caso, else elije
        clasificar los temas en 5 diferentes categorias pero el número puede variar de acuerdo al caso. 
        Existen formas de medir la calidad del modelo, calculando lo que se denomina perplexity pero no se va
        a utilizar en este caso.
        """
        n_topics = 5
        df_atenciones_filtro = self.filter()
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(2,3), min_df=100, max_df=0.85)),
            ('lda', LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online'))
        ])
        t0 = time()
        lda_model = text_pipeline.fit_transform(df_atenciones_filtro['hechos_filtrados'])
        print(time() - t0)
        tfidf = text_pipeline.steps[0][1]
        lda = text_pipeline.steps[1][1]
        vocabulario = tfidf.get_feature_names()
        top_topics = 5
        topic_dict = {}
        topic_scores = []
        for topic_idx, topic in enumerate(lda.components_):
            topic_dict[str(topic_idx)] = ",".join([vocabulario[i] for i in topic.argsort()[:-top_topics - 1:-1]])
            topic_scores.append([topic[i] for i in topic.argsort()[:-top_topics - 1:-1]])
        df_topics_lda = pd.DataFrame(topic_dict, index=['bigrams'])
        df_topics_lda = df_topics_lda.T.reset_index()
        df_topics_names = pd.DataFrame(df_topics_lda.bigrams.str.split(',').tolist(), index=df_topics_lda.index) \
                    .stack() \
                    .reset_index() \
                    .drop(['level_1'], axis=1) \
                    .rename(columns={0:'bigrams', 'level_0':'topic'})
        df_topics_scores = pd.DataFrame(topic_scores) \
                .stack() \
                .reset_index(drop=True)
        df_topics = pd.concat([df_topics_names, df_topics_scores], axis=1) \
                .rename(columns={0:'score'})
        plt.show()
        fig = plt.figure(figsize=(12,10))
        for i in range(1,n_topics+1):    
            plt.subplot(3,2,i,frameon=True)
            sns.barplot(x='score', y='bigrams', data=df_topics[df_topics['topic']==i-1], orient='h')
            plt.title("Topico {}".format(i))
            plt.xlabel('')
            plt.ylabel('')
            plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor())
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        topicos_html = 'data:image/png;base64,{}'.format(base64_image)
        return topicos_html