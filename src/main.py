from flask import Flask,render_template
from utils.explorer import explorer
import nltk
from nltk.corpus import stopwords

app= Flask(__name__)

nltk.download('stopwords')
stopwords_es = stopwords.words('spanish')

@app.route('/')
def home():
        return render_template('index.html')

@app.route('/wordcloud/')
def show_word_cloud():
        explorer_service = explorer(stopwords_es)
        wordcloud = explorer_service.generate_word_cloud()
        return render_template('wordcloud.html',wordcloud_image=wordcloud)

@app.route('/trigramas/')
def show_trigramas():
        explorer_service = explorer(stopwords_es)
        trigramas = explorer_service.generate_trigramas()
        return render_template('trigramas.html',trigramas_image=trigramas)

@app.route('/comunas/')
def show_comunas():
        explorer_service = explorer(stopwords_es)
        comunas = explorer_service.generate_comunas()
        return render_template('comunas.html',comunas_image=comunas)

@app.route('/municipios/')
def show_municipios():
        explorer_service = explorer(stopwords_es)
        municipios = explorer_service.generate_municipios()
        return render_template('municipios.html',municipios_image=municipios)

if __name__ == '__main__':
        app.run(debug=True)