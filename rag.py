"""
Implementación del sistema RAG (Retrieval-Augmented Generation).

Este módulo proporciona las funciones principales para cargar los documentos con embeddings,
realizar búsquedas de documentos relevantes y generar respuestas con y sin contexto
para permitir la comparación entre ambos enfoques.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import time

from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from dotenv import load_dotenv

# Import Anthropic integration for Haystack
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Verificar la presencia de las API keys
if "ANTHROPIC_API_KEY" not in os.environ:
    logger.warning("No se encontró ANTHROPIC_API_KEY en el entorno.")
    logger.warning("Por favor, crea un archivo .env con tu API key de Anthropic.")

if "OPENAI_API_KEY" not in os.environ:
    logger.warning("No se encontró OPENAI_API_KEY en el entorno.")
    logger.warning("Por favor, crea un archivo .env con tu API key de OpenAI.")

# Directorio de vectordb
VECTORDB_DIR = "vectordb"

# Modelos predeterminados por proveedor
MODELOS_DEFAULT = {
    "anthropic": "claude-3-haiku-20240307",
    "openai": "gpt-4o-mini"
}

class SistemaRAG:
    """Clase principal que implementa la funcionalidad RAG."""
    
    def __init__(self, provider: str = "anthropic", modelo_llm: str = None):
        """
        Inicializa el sistema RAG.
        
        Args:
            provider: Proveedor LLM a utilizar ("anthropic" o "openai").
            modelo_llm: Modelo específico a utilizar. Si es None, se usa el predeterminado para el proveedor.
        """
        self.provider = provider.lower()
        self.modelo_llm = modelo_llm or MODELOS_DEFAULT.get(self.provider, MODELOS_DEFAULT["anthropic"])
        self.documento_store = self._cargar_documento_store()
        
        # Inicializar componentes
        self.text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        # Configurar retriever con parámetros válidos
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.documento_store,
            return_embedding=True,  # Asegura que se devuelvan embeddings completos
            top_k=10  # Valor predeterminado
        )
        
        # Inicializar pipelines
        self._configurar_pipelines()
        
        logger.info(f"Sistema RAG inicializado con modelo {modelo_llm}")
        logger.info(f"Document store cargado con {len(self.documento_store.filter_documents())} documentos")
    
    def _cargar_documento_store(self) -> InMemoryDocumentStore:
        """
        Carga el document store desde el directorio vectordb utilizando archivos JSON.
        
        Returns:
            InMemoryDocumentStore: Document store cargado.
        """
        ruta_documentos = os.path.join(VECTORDB_DIR, "documentos_con_embeddings.json")
        
        if not os.path.exists(ruta_documentos):
            logger.warning(f"No se encontraron documentos en {ruta_documentos}")
            logger.warning("Iniciando document store vacío. Ejecuta indexador.py para generar embeddings.")
            return InMemoryDocumentStore()
        
        try:
            # Cargar los documentos con embeddings desde JSON
            with open(ruta_documentos, 'r', encoding='utf-8') as f:
                documentos_dict = json.load(f)
            
            # Crear un nuevo DocumentStore
            document_store = InMemoryDocumentStore()
            
            # Reconstruir los documentos de Haystack
            documentos = []
            for doc_dict in documentos_dict:
                # Reconstruir el embedding como array de numpy si existe
                embedding = None
                if doc_dict.get("embedding") is not None:
                    embedding = np.array(doc_dict["embedding"], dtype=np.float32)
                
                # Crear el documento de Haystack
                documento = Document(
                    content=doc_dict["content"],
                    meta=doc_dict["meta"],
                    embedding=embedding
                )
                documentos.append(documento)
            
            # Agregar los documentos al document store
            if documentos:
                document_store.write_documents(documentos)
                logger.info(f"Se cargaron {len(documentos)} documentos al Document Store")
            
            return document_store
            
        except Exception as e:
            logger.error(f"Error al cargar los documentos: {str(e)}")
            logger.error("Iniciando document store vacío")
            return InMemoryDocumentStore()
    
    def _configurar_pipelines(self) -> None:
        """Configura los pipelines de Haystack para RAG."""
        # Pipeline para recuperar documentos
        self.retrieval_pipeline = Pipeline()
        self.retrieval_pipeline.add_component("text_embedder", self.text_embedder)
        self.retrieval_pipeline.add_component("retriever", self.retriever)
        self.retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        
        # Plantilla para RAG
        template_rag = [
            ChatMessage.from_system(
                """Eres un experto en marketing con amplio conocimiento del sector. 
                Tu tarea es proporcionar respuestas precisas y detalladas basadas EXCLUSIVAMENTE 
                en la información proporcionada en los documentos de contexto.
                
                INSTRUCCIONES IMPORTANTES:
                1. Cita explícitamente datos y ejemplos de los documentos proporcionados
                2. Estructura tu respuesta incluyendo datos concretos del contexto
                3. Menciona los títulos o referencias de los documentos que utilizas en tu respuesta
                4. Si la información no está en los documentos, indícalo claramente
                5. NO utilices conocimiento general que no esté en los documentos
                """
            ),
            ChatMessage.from_user(
                """
                Documentos de contexto:
                {% for documento in documentos %}
                --- DOCUMENTO: {{ documento.meta.titulo if documento.meta and documento.meta.titulo else 'Documento sin título' }} ---
                {{ documento.content }}
                
                {% endfor %}
                
                Pregunta: {{pregunta}}
                
                IMPORTANTE: Responde de forma completa y detallada basándote ÚNICAMENTE en la información de los documentos proporcionados. Cita partes específicas de los documentos para respaldar tu respuesta.
                """
            )
        ]
        
        # Plantilla para LLM sin contexto
        template_llm = [
            ChatMessage.from_system(
                """Eres un experto en marketing con amplio conocimiento del sector.
                Tu tarea es proporcionar respuestas precisas y detalladas a las preguntas sobre marketing.
                Si no conoces la respuesta o no tienes suficiente información, indícalo claramente."""
            ),
            ChatMessage.from_user(
                """
                Pregunta: {{pregunta}}
                
                Responde de forma completa y detallada.
                """
            )
        ]
        
        # Crear generadores según el proveedor
        if self.provider == "anthropic":
            # Verificar API key
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("No se encontró ANTHROPIC_API_KEY en el entorno. Configura esta variable en el archivo .env")
            
            rag_generator = AnthropicChatGenerator(model=self.modelo_llm)
            llm_generator = AnthropicChatGenerator(model=self.modelo_llm)
            logger.info(f"Usando generador Anthropic con modelo {self.modelo_llm}")
        else:  # openai
            # Verificar API key
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("No se encontró OPENAI_API_KEY en el entorno. Configura esta variable en el archivo .env")
            
            rag_generator = OpenAIChatGenerator(model=self.modelo_llm)
            llm_generator = OpenAIChatGenerator(model=self.modelo_llm)
            logger.info(f"Usando generador OpenAI con modelo {self.modelo_llm}")
        
        # Pipeline para RAG
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template_rag))
        self.rag_pipeline.add_component("llm", rag_generator)
        self.rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
        
        # Pipeline para LLM sin contexto
        self.llm_pipeline = Pipeline()
        self.llm_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template_llm))
        self.llm_pipeline.add_component("llm", llm_generator)
        self.llm_pipeline.connect("prompt_builder.prompt", "llm.messages")
    
    def recuperar_documentos(self, pregunta: str, top_k: int = 3) -> List[Document]:
        """
        Recupera los documentos más relevantes para una pregunta.
        
        Args:
            pregunta: Pregunta del usuario.
            top_k: Número de documentos a recuperar.
            
        Returns:
            List[Document]: Lista de documentos relevantes.
        """
        inicio = time.time()
        
        try:
            # Imprimir información sobre el DocumentStore antes de la recuperación
            print("\n==== DEBUG: DOCUMENTO STORE ANTES DE RECUPERACIÓN ====")
            docs_total = self.documento_store.filter_documents()
            print(f"Total de documentos en DocumentStore: {len(docs_total)}")
            if docs_total:
                print("Ejemplo de metadatos del primer documento:")
                print(f"Tipo: {type(docs_total[0])}")
                print(f"Meta: {docs_total[0].meta}")
                print(f"Tiene atributo 'titulo' en meta: {'titulo' in docs_total[0].meta}")
            
            # Ejecutar recuperación
            resultado = self.retrieval_pipeline.run({"text_embedder": {"text": pregunta}, 
                                                    "retriever": {"top_k": top_k}})
            # Acceder correctamente a los documentos dentro del resultado
            documentos = resultado["retriever"]["documents"]
            
            # Convertir documentos si son strings
            documentos_procesados = []
            for i, doc in enumerate(documentos):
                if isinstance(doc, str):
                    # Para strings, simplemente crear un nuevo Document
                    doc_procesado = Document(
                        content=doc,
                        meta={"titulo": f"Documento {i+1}"}
                    )
                    documentos_procesados.append(doc_procesado)
                    print(f"Documento {i+1} convertido de string a Document")
                else:
                    documentos_procesados.append(doc)
            
            # Usar los documentos procesados
            documentos = documentos_procesados
            
            # Imprimir información sobre los documentos recuperados
            print("\n==== DEBUG: DOCUMENTOS RECUPERADOS ====")
            print(f"Documentos recuperados: {len(documentos)}")
            for i, doc in enumerate(documentos):
                print(f"\nDocumento {i+1}:")
                print(f"Tipo: {type(doc)}")
                print(f"Tiene atributo meta: {hasattr(doc, 'meta')}")
                if hasattr(doc, 'meta'):
                    print(f"Meta: {doc.meta}")
                    print(f"Meta tipo: {type(doc.meta)}")
                    print(f"Tiene 'titulo' en meta: {'titulo' in doc.meta}")
                    if 'titulo' in doc.meta:
                        print(f"Valor de 'titulo': {doc.meta['titulo']}")
                print(f"Tiene atributo content: {hasattr(doc, 'content')}")
                if hasattr(doc, 'content'):
                    print(f"Content (primeros 50 caracteres): {doc.content[:50]}...")
            
            fin = time.time()
            logger.info(f"Recuperados {len(documentos)} documentos en {fin-inicio:.2f} segundos")
            
            return documentos
        except Exception as e:
            logger.error(f"Error al recuperar documentos: {str(e)}")
            print(f"\n==== DEBUG: EXCEPCIÓN EN RECUPERACIÓN ====")
            print(f"Error: {str(e)}")
            return []
    
    def generar_respuesta_con_contexto(self, pregunta: str, documentos: List[Union[Document, str]]) -> Tuple[str, float]:
        """
        Genera una respuesta utilizando RAG.
        
        Args:
            pregunta: Pregunta del usuario.
            documentos: Documentos de contexto.
            
        Returns:
            Tuple[str, float]: (Respuesta generada, tiempo de generación).
        """
        inicio = time.time()
        
        try:
            # Asegurar que todos los documentos sean objetos Document
            documentos_procesados = []
            for i, doc in enumerate(documentos):
                if isinstance(doc, str):
                    # Convertir string a Document con metadatos básicos
                    doc_procesado = Document(
                        content=doc,
                        meta={"titulo": f"Documento {i+1}"}
                    )
                    documentos_procesados.append(doc_procesado)
                    print(f"Documento {i+1} convertido de string a Document")
                else:
                    documentos_procesados.append(doc)
            
            resultado = self.rag_pipeline.run({"prompt_builder": {"pregunta": pregunta, 
                                                                "documentos": documentos_procesados}})
            respuesta = resultado["llm"]["replies"][0].text
            
            fin = time.time()
            tiempo = fin - inicio
            logger.info(f"Respuesta RAG generada en {tiempo:.2f} segundos")
            
            return respuesta, tiempo
        except Exception as e:
            logger.error(f"Error al generar respuesta RAG: {str(e)}")
            return "Error al generar respuesta.", 0
    
    def generar_respuesta_sin_contexto(self, pregunta: str) -> Tuple[str, float]:
        """
        Genera una respuesta sin utilizar RAG.
        
        Args:
            pregunta: Pregunta del usuario.
            
        Returns:
            Tuple[str, float]: (Respuesta generada, tiempo de generación).
        """
        inicio = time.time()
        
        try:
            resultado = self.llm_pipeline.run({"prompt_builder": {"pregunta": pregunta}})
            respuesta = resultado["llm"]["replies"][0].text
            
            fin = time.time()
            tiempo = fin - inicio
            logger.info(f"Respuesta LLM generada en {tiempo:.2f} segundos")
            
            return respuesta, tiempo
        except Exception as e:
            logger.error(f"Error al generar respuesta LLM: {str(e)}")
            return "Error al generar respuesta.", 0
    
    def generar_respuesta_completa(self, 
                                  pregunta: str, 
                                  top_k: int = 3) -> Dict[str, Any]:
        """
        Genera respuestas tanto con RAG como sin contexto para una pregunta.
        
        Args:
            pregunta: Pregunta del usuario.
            top_k: Número de documentos a recuperar.
            
        Returns:
            Dict: Diccionario con ambas respuestas y documentos relevantes.
        """
        # Recuperar documentos relevantes
        documentos = self.recuperar_documentos(pregunta, top_k)
        
        # Generar respuesta RAG con contexto
        respuesta_rag, tiempo_rag = self.generar_respuesta_con_contexto(pregunta, documentos)
        
        # Generar respuesta tradicional sin contexto
        respuesta_tradicional, tiempo_llm = self.generar_respuesta_sin_contexto(pregunta)
        
        return {
            "respuesta_rag": respuesta_rag,
            "tiempo_rag": tiempo_rag,
            "respuesta_tradicional": respuesta_tradicional,
            "tiempo_llm": tiempo_llm,
            "documentos": documentos,
            "num_documentos": len(documentos)
        }

# Ejemplos predefinidos para demostración
EJEMPLOS_PREDEFINIDOS = [
    "¿Qué es NeuroBrand y cómo funciona?",
    "¿Qué es QuantumMarket y qué capacidades principales ofrece?",
    "Explica el Método Triaxial de Posicionamiento",
    "¿Cuáles son los componentes del Índice de Permeabilidad Cultural (IPC)?",
    "¿Qué es el Marketing Inmersivo Biométrico y qué consideraciones éticas plantea?",
    "¿Qué es el Micro-targeting Cronobiológico y qué resultados ha generado?",
    "¿Cómo implementó Banco Ibérico QuantumMarket y qué resultados obtuvo?",
    "¿Cómo revitalizó Café Tradición su marca utilizando NeuroBrand?",
    "¿Qué métricas propias introduce la Dinámica de Convergencia Multicanal?",
    "¿Qué es RetailMapper y qué métricas exclusivas introduce?"
]

# Inicialización para pruebas
if __name__ == "__main__":
    # Ejemplo de uso
    sistema_rag = SistemaRAG()
    pregunta = "¿Qué es MarketPulse y para qué sirve?"
    
    resultado = sistema_rag.generar_respuesta_completa(pregunta)
    
    print("\n=== RESPUESTA RAG ===")
    print(resultado["respuesta_rag"])
    
    print("\n=== RESPUESTA LLM ===")
    print(resultado["respuesta_tradicional"])
    
    print("\n=== DOCUMENTOS UTILIZADOS ===")
    for i, doc in enumerate(resultado["documentos"]):
        print(f"Documento {i+1}: {doc.meta.get('titulo', 'Sin título')}")
