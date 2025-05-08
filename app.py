"""
App principal de demo RAG para marketing.

Esta aplicación Streamlit demuestra las ventajas de usar RAG (Retrieval-Augmented Generation)
sobre LLMs tradicionales, con una interfaz sencilla que permite al usuario enviar preguntas
y ver la comparación entre ambos enfoques lado a lado.
"""

import os
import streamlit as st
import time
from dotenv import load_dotenv

# Importar módulos propios
from rag import SistemaRAG, EJEMPLOS_PREDEFINIDOS

# Configurar la página
st.set_page_config(
    page_title="Demo RAG para Marketing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar variables de entorno
load_dotenv()

# Verificar qué proveedores están disponibles basado en las API keys existentes
available_providers = []
provider_mapping = {
    "Anthropic Claude": {
        "key": "anthropic",
        "env_var": "ANTHROPIC_API_KEY"
    },
    "OpenAI GPT": {
        "key": "openai",
        "env_var": "OPENAI_API_KEY"
    }
}

# Comprobar cada proveedor
for display_name, config in provider_mapping.items():
    if config["env_var"] in os.environ and os.environ[config["env_var"]]:
        available_providers.append(display_name)

# Si no hay proveedores disponibles, mostrar error
if not available_providers:
    st.error("⚠️ No se ha encontrado ninguna API key de proveedor LLM en el entorno.")
    st.info("""Por favor, crea un archivo llamado `.env` en el directorio raíz con al menos una de las siguientes claves API:
    ```
    ANTHROPIC_API_KEY=tu_clave_api_aquí
    OPENAI_API_KEY=tu_clave_api_aquí
    ```""")
    st.stop()

# Barra lateral para opciones de configuración
with st.sidebar:
    st.title("Configuración")
    
    # Solo mostrar el selector si hay más de un proveedor disponible
    if len(available_providers) > 1:
        provider = st.radio(
            "Seleccionar Proveedor LLM:",
            available_providers,
            index=0,
            help="Elige el proveedor de LLM que deseas utilizar"
        )
    else:
        # Si solo hay un proveedor, mostrarlo sin opción de selección
        provider = available_providers[0]
        st.info(f"Usando el único proveedor disponible: {provider}")
    
    # Convertir la selección amigable a la clave interna
    provider_key = provider_mapping[provider]["key"]

# Función para inicializar el sistema RAG
@st.cache_resource
def inicializar_sistema_rag(provider):
    """
    Inicializa y devuelve una instancia del sistema RAG.
    
    Args:
        provider: El proveedor a utilizar ('anthropic' o 'openai')
    """
    with st.spinner(f"Inicializando el sistema RAG con {provider}..."):
        sistema = SistemaRAG(provider=provider)
    return sistema

# Función para formatear el tiempo
def formatear_tiempo(segundos):
    """Formatea el tiempo en segundos a un formato legible."""
    if segundos < 0.1:
        return f"{segundos*1000:.0f} ms"
    else:
        return f"{segundos:.2f} s"

# Inicializar el sistema RAG
sistema_rag = inicializar_sistema_rag(provider_key)

# Título y descripción
st.title("🔍 Demo RAG para Marketing")

st.markdown("""
Esta aplicación demuestra el poder de **Retrieval-Augmented Generation (RAG)** para mejorar 
las respuestas de LLMs, especialmente cuando se trata de información especializada o que 
no está en su entrenamiento base.

Observa lado a lado cómo las respuestas generadas con RAG son más precisas y contienen menos 
alucinaciones que las generadas por un LLM tradicional.
""")

# Sección de entrada
st.header("📝 Tu pregunta")

# Contenedor para ejemplos predefinidos y entrada de texto
entrada_container = st.container()

with entrada_container:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        pregunta_usuario = st.text_area(
            "Escribe tu pregunta aquí:",
            height=100,
            placeholder="Por ejemplo: ¿Qué es NeuroBrand y cómo funciona?"
        )
    
    with col2:
        st.write("**Ejemplos predefinidos**")
        ejemplo_seleccionado = st.selectbox(
            "Selecciona un ejemplo:",
            options=["Seleccionar ejemplo..."] + EJEMPLOS_PREDEFINIDOS,
            index=0
        )
        
        if ejemplo_seleccionado != "Seleccionar ejemplo...":
            pregunta_usuario = ejemplo_seleccionado
            st.info(f"Ejemplo seleccionado: {ejemplo_seleccionado}")

# Botón para enviar
enviar = st.button("🚀 Enviar pregunta", type="primary")

# Procesamiento de la pregunta
if enviar and pregunta_usuario:
    # Mostrar spinner durante la generación
    with st.spinner("Generando respuestas..."):
        inicio = time.time()
        resultado = sistema_rag.generar_respuesta_completa(pregunta_usuario)
        tiempo_total = time.time() - inicio
    
    # Extraer respuestas y metadatos
    respuesta_rag = resultado["respuesta_rag"]
    respuesta_llm = resultado["respuesta_tradicional"]
    tiempo_rag = resultado["tiempo_rag"]
    tiempo_llm = resultado["tiempo_llm"]
    documentos = resultado["documentos"]
    
    # Sección de comparación
    st.header("🔄 Comparación de respuestas")
    st.write(f"Tiempo total: {formatear_tiempo(tiempo_total)}")
    
    # Crear columnas para comparación
    col1, col2 = st.columns(2)
    
    # Estilo para las tarjetas
    estilo_tarjeta = """
    <style>
    .tarjeta {
        border-radius: 5px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    .tarjeta-rag {
        background-color: #e7f0fd;
        border-left: 5px solid #3b82f6;
    }
    .tarjeta-llm {
        background-color: #f5f5f5;
        border-left: 5px solid #6b7280;
    }
    .titulo-tarjeta {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .contenido-tarjeta {
        font-size: 1rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .metadata {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 10px;
    }
    </style>
    """
    
    st.markdown(estilo_tarjeta, unsafe_allow_html=True)
    
    # Columna para respuesta RAG
    with col1:
        st.markdown(f"""
        <div class="tarjeta tarjeta-rag">
            <div class="titulo-tarjeta">🔍 Respuesta con RAG</div>
            <div class="contenido-tarjeta">{respuesta_rag}</div>
            <div class="metadata">Tiempo de generación: {formatear_tiempo(tiempo_rag)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Columna para respuesta LLM tradicional
    with col2:
        st.markdown(f"""
        <div class="tarjeta tarjeta-llm">
            <div class="titulo-tarjeta">🤖 Respuesta sin RAG</div>
            <div class="contenido-tarjeta">{respuesta_llm}</div>
            <div class="metadata">Tiempo de generación: {formatear_tiempo(tiempo_llm)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sección de documentos recuperados
    if documentos:
        st.header("📄 Documentos utilizados")
        st.write(f"Se utilizaron {len(documentos)} documentos para generar la respuesta RAG:")
        
        # Estilo CSS para los documentos
        st.markdown("""
        <style>
        .documento-titulo {
            font-size: 1.2rem;
            font-weight: bold;
            padding: 10px 15px;
            background-color: #f8f9fa;
            border-left: 5px solid #4caf50;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .documento-contenido {
            padding: 0 15px;
            border-left: 1px solid #e0e0e0;
            margin-left: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        for i, doc in enumerate(documentos):
            # Verificar que sea un objeto Document con atributos meta y content
            if hasattr(doc, 'meta') and doc.meta is not None:
                titulo = doc.meta.get('titulo', f'Documento {i+1}')
            else:
                titulo = f'Documento {i+1}'
            
            # Mostrar título del documento con estilo
            st.markdown(f'<div class="documento-titulo">📑 {titulo}</div>', unsafe_allow_html=True)
            
            # Mostrar contenido del documento con estilo
            with st.expander("Ver documento completo", expanded=True):
                if hasattr(doc, 'content'):
                    st.markdown(f'<div class="documento-contenido">{doc.content}</div>', unsafe_allow_html=True)
                else:
                    # Si no es un Document, convertir a string
                    st.markdown(f'<div class="documento-contenido">{str(doc)}</div>', unsafe_allow_html=True)
    else:
        st.warning("No se encontraron documentos relevantes para esta consulta.")

# Información adicional
with st.sidebar:
    st.title("Sobre esta demo")
    
    st.markdown("""
    ## ✨ Beneficios de RAG demostrados
    
    1. **Mayor precisión factual**: RAG proporciona respuestas basadas en la documentación específica
    2. **Menor alucinación**: El modelo tiene menos tendencia a inventar información
    3. **Transparencia**: Se muestran las fuentes utilizadas para generar la respuesta
    4. **Información específica**: Acceso a datos que no están en el entrenamiento del LLM
    """)
    
    st.markdown("""
    ## 🔧 Tecnologías utilizadas
    
    - **Python**: Lenguaje base
    - **Streamlit**: Interfaz de usuario
    - **Haystack**: Framework para RAG
    - **LLMs**: Soporte para Anthropic Claude y OpenAI GPT
    """)
    
    st.markdown("""
    ## ❓ ¿Cómo funciona?
    
    1. **Datos**: Generamos datos sintéticos de marketing
    2. **Indexación**: Procesamos documentos y creamos embeddings
    3. **Recuperación**: Buscamos documentos relevantes a la consulta
    4. **Generación**: Usamos el mismo modelo para dos tareas:
       - Generar respuestas con contexto recuperado (RAG)
       - Generar respuestas sin contexto adicional (LLM básico)
    5. **Comparación**: Presentamos ambas respuestas lado a lado
    """)

# Footer
st.markdown("---")
st.markdown("🎓 Demo RAG + Marketing by Farid")
