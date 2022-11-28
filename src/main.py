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

@app.route('/subtemas/')
def show_subtemas():
        explorer_service = explorer(stopwords_es)
        subtemas = explorer_service.generate_subtemas()
        return render_template('subtemas.html',subtemas_image=subtemas)

@app.route('/temas/')
def show_temas():
        explorer_service = explorer(stopwords_es)
        temas = explorer_service.generate_temas()
        return render_template('temas.html',temas_image=temas)

@app.route('/servicios/')
def show_servicios():
        explorer_service = explorer(stopwords_es)
        servicios = explorer_service.generate_servicios()
        return render_template('servicios.html',servicios_image=servicios)

@app.route('/usuarios/')
def show_usuarios():
        explorer_service = explorer(stopwords_es)
        grafico, tabla,bar = explorer_service.generate_usuarios()
        return render_template('usuarios.html',grafico_image=grafico,
                tabla_image=tabla, bar_image=bar)

@app.route('/eventos/')
def show_eventos():
        explorer_service = explorer(stopwords_es)
        dia, sura, eps, savia = explorer_service.generate_eventos()
        return render_template('eventos.html', dia_image=dia, 
                sura_image=sura, eps_image=eps,
                savia_image=savia)

@app.route('/topicos/')
def show_topicos():
        explorer_service = explorer(stopwords_es)
        topicos = explorer_service.generate_topicos()
        return render_template('topicos.html',topicos_image=topicos)

if __name__ == '__main__':
        app.run(debug=True)