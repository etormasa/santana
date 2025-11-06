# api/index.py
# Vercel necesita que exportes una WSGI app llamada `app`.
# Reutilizamos tu Flask de app.py.
from app import app as app  # NO cambies el nombre a "application"; Vercel espera "app"
