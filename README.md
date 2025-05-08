# Demo RAG para Marketing

Esta aplicación demuestra el poder de **Retrieval-Augmented Generation (RAG)** para mejorar las respuestas de LLMs, especialmente cuando se trata de información especializada o que no está en su entrenamiento base.

## 🚀 Características

- Compara respuestas de un modelo RAG vs un LLM tradicional
- Muestra los documentos recuperados para la generación de respuestas
- Interfaz intuitiva con Streamlit
- Ejemplos predefinidos para demostrar los beneficios de RAG
- Utiliza datos sintéticos de marketing con información ficticia pero plausible

## ✨ Beneficios de RAG demostrados

1. **Mayor precisión factual**: RAG proporciona respuestas basadas en la documentación específica
2. **Menor alucinación**: El modelo tiene menos tendencia a inventar información
3. **Transparencia**: Se muestran las fuentes utilizadas para generar la respuesta
4. **Información específica**: Acceso a datos que no están en el entrenamiento del LLM

## 🔧 Requisitos

- Python 3.12
- Una clave API de Anthropic y/o OpenAI (según el proveedor que quieras usar)

## 📋 Instalación

1. Clona este repositorio

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Configura tus claves API en el archivo `.env`:
   ```bash
   cp .env.template .env
   # Edita el archivo .env con las claves API de los proveedores que vayas a usar
   ```

## 🚀 Ejecución

Para ejecutar la aplicación, sigue estos pasos:

1. Procesa los documentos y crea embeddings:
   ```bash
   python indexador.py
   ```

2. Ejecuta la aplicación Streamlit:
   ```bash
   streamlit run app.py
   ```

3. Abre tu navegador en: http://localhost:8501

## 📁 Estructura del proyecto

- `app.py`: Aplicación principal con interfaz Streamlit
- `rag.py`: Core RAG logic, invocado desde `app.py`
- `indexador.py`: Procesamiento y creación de embeddings para documentos
- `requirements.txt`: Dependencias del proyecto
- `.env.template`: Plantilla para configuración de API keys
- `data/`: Directorio que contiene documentos JSON prefabricados con información de marketing
  - `documentos_marketing.json`: Archivo combinado con todos los documentos
  - `doc_001.json` a `doc_020.json`: Documentos individuales con temas de marketing específicos

## How it works

1. **Data**: La aplicación utiliza documentos prefabricados con información ficticia de marketing (conceptos inventados, metodologías ficticias, casos de estudio)
2. **Indexing**: Procesa los documentos JSON y crea embeddings con SentenceTransformers
3. **Retrieval**: Dada una consulta, encuentra los documentos más relevantes
4. **Generation**: Utiliza el mismo modelo LLM (Anthropic o OpenAI) para dos tareas:
   - Generar respuestas con contexto recuperado (RAG)
   - Generar respuestas sin contexto adicional (LLM básico)
5. **Comparison**: Presenta ambas respuestas lado a lado y muestra los documentos utilizados

## 🔄 Selección de proveedor LLM

La aplicación permite elegir entre dos proveedores de modelos de lenguaje:

- **Anthropic Claude**: Modelos Claude optimizados para tareas de generación de texto
- **OpenAI GPT**: Modelos GPT con capacidades avanzadas de procesamiento de lenguaje

Simplemente selecciona tu proveedor preferido en la barra lateral de la aplicación. La aplicación utilizará la clave API correspondiente configurada en tu archivo `.env`.

## Consideraciones de implementación

- Los datos son sintéticos y creados específicamente para demostrar la diferencia entre RAG y LLMs tradicionales.
- La aplicación mantiene el almacén de documentos en memoria.
- Los modelos predeterminados son `claude-3-haiku-20240307` para Anthropic y `gpt-4o-mini` para OpenAI, pero pueden modificarse en el código.
