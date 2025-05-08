"""
Indexador de documentos de marketing.

Este script procesa los documentos generados por datos_marketing.py, 
los convierte en objetos Document de Haystack, genera embeddings, 
y los guarda en un formato persistente en el directorio vectordb/.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import pickle
import time
from datetime import datetime

from tqdm import tqdm
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Directorios de datos
DATA_DIR = "data"
VECTORDB_DIR = "vectordb"

# Asegurar que los directorios existan
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)

def cargar_documentos() -> List[Dict[str, Any]]:
    """
    Carga los documentos JSON del directorio de datos.
    
    Returns:
        List[Dict]: Lista de documentos en formato de diccionario.
    """
    ruta_coleccion = os.path.join(DATA_DIR, "documentos_marketing.json")
    
    if not os.path.exists(ruta_coleccion):
        logger.error(f"No se encontró el archivo de documentos en {ruta_coleccion}")
        raise FileNotFoundError(f"El archivo {ruta_coleccion} no existe.")
    
    try:
        print(f"Cargando documentos desde {ruta_coleccion}:")
        inicio = time.time()
        
        with tqdm(total=1, desc="Cargando JSON", unit="archivo") as pbar:
            with open(ruta_coleccion, 'r', encoding='utf-8') as f:
                documentos = json.load(f)
                pbar.update(1)
        
        fin = time.time()
        print(f"✅ Se cargaron {len(documentos)} documentos en {fin-inicio:.2f} segundos")
        logger.info(f"Se cargaron {len(documentos)} documentos de {ruta_coleccion}")
        return documentos
    except Exception as e:
        logger.error(f"Error al cargar los documentos: {str(e)}")
        raise

def convertir_a_haystack_documents(documentos_raw: List[Dict[str, Any]]) -> List[Document]:
    """
    Convierte los documentos crudos en objetos Document de Haystack.
    
    Args:
        documentos_raw: Lista de documentos en formato diccionario.
        
    Returns:
        List[Document]: Lista de documentos de Haystack.
    """
    documentos_haystack = []
    
    print("Convirtiendo documentos al formato Haystack:")
    for doc in tqdm(documentos_raw, desc="Convirtiendo documentos", unit="doc"):
        # Extraer contenido y metadatos
        contenido = doc["contenido"]
        meta = {
            "id": doc["id"],
            "titulo": doc["titulo"],
            "fecha_creacion": doc["fecha_creacion"]
        }
        
        # Crear documento de Haystack
        documento = Document(content=contenido, meta=meta)
        documentos_haystack.append(documento)
    
    logger.info(f"Se convirtieron {len(documentos_haystack)} documentos al formato Haystack")
    return documentos_haystack

def generar_embeddings(documentos: List[Document], 
                      modelo: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[Document]:
    """
    Genera embeddings para los documentos.
    
    Args:
        documentos: Lista de documentos de Haystack.
        modelo: Nombre del modelo de embeddings a utilizar.
        
    Returns:
        List[Document]: Documentos con embeddings generados.
    """
    logger.info(f"Generando embeddings con el modelo {modelo}")
    print(f"Generando embeddings para {len(documentos)} documentos:")
    
    # Mostrar una barra de progreso para la carga del modelo
    print("Inicializando el modelo de embeddings:")
    with tqdm(total=1, desc="Cargando modelo", unit="modelo") as pbar:
        # Inicializar el embedder
        embedder = SentenceTransformersDocumentEmbedder(model=modelo)
        embedder.warm_up()
        pbar.update(1)
    
    # Medir el tiempo de generación
    inicio = time.time()
    
    # Crear una barra de progreso para visualizar el proceso
    with tqdm(total=100, desc="Procesando embeddings", unit="%") as pbar:
        # Generar embeddings - como el proceso interno no es iterable,
        # simularemos el progreso en etapas
        pbar.update(10)  # Inicio del proceso
        
        resultado = embedder.run(documentos)
        pbar.update(90)  # Completado
        
        docs_con_embeddings = resultado["documents"]
    
    fin = time.time()
    tiempo_total = fin - inicio
    
    logger.info(f"Se generaron embeddings para {len(docs_con_embeddings)} documentos en {tiempo_total:.2f} segundos")
    print(f"✅ Embeddings generados en {tiempo_total:.2f} segundos")
    
    return docs_con_embeddings

def guardar_vectordb(documentos: List[Document]) -> None:
    """
    Guarda los documentos con embeddings en un formato persistente.
    
    Args:
        documentos: Documentos de Haystack con embeddings.
    """
    print(f"Guardando {len(documentos)} documentos en la base de vectores:")
    inicio = time.time()
    
    # Crear document store y barra de progreso
    with tqdm(total=3, desc="Guardando vectordb", unit="paso") as pbar:
        # Paso 1: Crear document store e insertar documentos
        document_store = InMemoryDocumentStore()
        document_store.write_documents(documentos)
        logger.info(f"Se guardaron {len(documentos)} documentos en el DocumentStore")
        pbar.update(1)
        
        # Paso 2: Guardar el estado del document store como diccionario serializable
        ruta_documento_store = os.path.join(VECTORDB_DIR, "document_store.json")
        try:
            document_store_dict = document_store.to_dict()
            with open(ruta_documento_store, 'w', encoding='utf-8') as f:
                json.dump(document_store_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Document store guardado en {ruta_documento_store}")
        except Exception as e:
            logger.error(f"Error al guardar document store como JSON: {str(e)}")
            # Intentar guardar solo la configuración básica
            configuracion = {
                "tipo": "InMemoryDocumentStore",
                "num_documentos": len(documentos),
                "tiempo_creacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(ruta_documento_store, 'w', encoding='utf-8') as f:
                json.dump(configuracion, f, ensure_ascii=False, indent=2)
        pbar.update(1)
        
        # Paso 3: Guardar los documentos como respaldo
        ruta_documentos = os.path.join(VECTORDB_DIR, "documentos_con_embeddings.json")
        documentos_serializables = []
        for doc in documentos:
            # Convertir embedding a lista si existe
            embedding = None
            if hasattr(doc, "embedding") and doc.embedding is not None:
                try:
                    # Convertir array de numpy a lista
                    embedding = doc.embedding.tolist()
                except:
                    # Si falla, intentar convertir directamente
                    embedding = list(doc.embedding) if doc.embedding is not None else None
            
            doc_dict = {
                "content": doc.content,
                "meta": doc.meta,
                "embedding": embedding
            }
            documentos_serializables.append(doc_dict)
            
        with open(ruta_documentos, 'w', encoding='utf-8') as f:
            json.dump(documentos_serializables, f, ensure_ascii=False, indent=2)
        pbar.update(1)
    
    fin = time.time()
    print(f"✅ Base de vectores guardada en {fin-inicio:.2f} segundos")
    logger.info(f"Documentos con embeddings guardados en {ruta_documentos}")

def procesar_documentos(modelo_embeddings: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    """
    Flujo completo de procesamiento e indexación de documentos.
    
    Args:
        modelo_embeddings: Modelo de embeddings a utilizar.
    """
    try:
        print("\n" + "="*50)
        print(" PROCESO DE INDEXACIÓN DE DOCUMENTOS DE MARKETING ")
        print("="*50 + "\n")
        
        # Medición del tiempo total
        tiempo_inicio_total = time.time()
        
        # Barra de progreso general del proceso
        etapas = ["Cargar documentos", "Convertir a formato Haystack", 
                 "Generar embeddings", "Guardar vectordb"]
        
        with tqdm(total=len(etapas), desc="Progreso general", 
                 unit="etapa", position=0) as pbar_general:
            
            # 1. Cargar documentos
            print("\n📄 ETAPA 1: Cargando documentos...")
            documentos_raw = cargar_documentos()
            pbar_general.update(1)
            
            # 2. Convertir a formato Haystack
            print("\n🔄 ETAPA 2: Convirtiendo documentos...")
            documentos_haystack = convertir_a_haystack_documents(documentos_raw)
            pbar_general.update(1)
            
            # 3. Generar embeddings
            print("\n🧠 ETAPA 3: Generando embeddings...")
            documentos_con_embeddings = generar_embeddings(documentos_haystack, modelo_embeddings)
            pbar_general.update(1)
            
            # 4. Guardar en vectordb
            print("\n💾 ETAPA 4: Guardando base de vectores...")
            guardar_vectordb(documentos_con_embeddings)
            pbar_general.update(1)
        
        # Calcular tiempo total
        tiempo_total = time.time() - tiempo_inicio_total
        
        print("\n" + "="*50)
        print(f"✅ PROCESO COMPLETADO EN {tiempo_total:.2f} SEGUNDOS")
        print(f"✅ Documentos procesados: {len(documentos_raw)}")
        print(f"✅ Vectores generados: {len(documentos_con_embeddings)}")
        print("="*50 + "\n")
        
        logger.info(f"Proceso de indexación completado con éxito en {tiempo_total:.2f} segundos")
    except Exception as e:
        logger.error(f"Error en el proceso de indexación: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    procesar_documentos()
