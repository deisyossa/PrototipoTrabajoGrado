# PrototipoTrabajoGrado
# Ejecucion proyecto
# 1. Instalación del virtualenv
- pip install virtualenv
# 2. Desactivar política de windows para permitir la activación de ejecución del entorno virtual
- Set-ExecutionPolicy Unrestricted -Scope CurrentUser
# 3. Activación del entrono virtual (Parado sobre la carpeta raiz)
- env/Scripts/activate
# 4. Instalación de librerías (Recomerdado instalarlas una vez se active el entorno virtual)
- pip install -r requirements.txt
# 5. Entrar a la carpeta src
- cd src
# Ejecutar el proyecto
- python main.py