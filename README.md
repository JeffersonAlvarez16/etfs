# ETF Investment Dashboard

Una aplicación web para analizar y dar seguimiento a inversiones en ETFs.

## Características

- Sistema de autenticación seguro
- Análisis de ETFs en tiempo real
- Seguimiento de inversiones
- Recomendaciones basadas en análisis técnico
- Visualización de datos y métricas

## Configuración Local

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar variables de entorno:
- Renombrar `.env.example` a `.env`
- Modificar las credenciales en el archivo `.env`:
```
ADMIN_USERNAME=tu_usuario
ADMIN_PASSWORD=tu_contraseña
```

3. Ejecutar la aplicación:
```bash
streamlit run etf_dashboard.py
```

## Opciones de Despliegue

### 1. Streamlit Cloud (Recomendado)

1. Crear una cuenta en [Streamlit Cloud](https://streamlit.io/cloud)
2. Conectar con tu repositorio de GitHub
3. Desplegar la aplicación desde el dashboard de Streamlit Cloud
4. Configurar las variables de entorno en la configuración del proyecto

### 2. Heroku

1. Crear una cuenta en [Heroku](https://heroku.com)
2. Instalar Heroku CLI
3. Crear un nuevo proyecto:
```bash
heroku create mi-etf-dashboard
```
4. Configurar variables de entorno:
```bash
heroku config:set ADMIN_USERNAME=tu_usuario
heroku config:set ADMIN_PASSWORD=tu_contraseña
```
5. Desplegar:
```bash
git push heroku main
```

### 3. DigitalOcean App Platform

1. Crear una cuenta en [DigitalOcean](https://digitalocean.com)
2. Crear una nueva App desde el dashboard
3. Conectar con tu repositorio
4. Configurar el comando de inicio: `streamlit run etf_dashboard.py`
5. Configurar variables de entorno en la sección de Settings

## Seguridad

- Las contraseñas se almacenan hasheadas usando bcrypt
- Usar contraseñas seguras en producción
- Mantener el archivo `.env` seguro y nunca compartirlo
- Cambiar las credenciales por defecto antes de desplegar

## Soporte

Para soporte o preguntas, crear un issue en el repositorio.
