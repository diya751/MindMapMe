import os
import json
import networkx as nx
from typing import List, Dict, Any
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import Config

class RAGProcessor:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name="llama3-8b-8192"
        )
        # Fast, CPU-friendly embeddings with prebuilt wheels (no torch)
        self.embeddings = FastEmbedEmbeddings()
        self.vectorstore = None
        self.documents = []
        self.concept_map = nx.DiGraph()
        self.flashcards = []
        
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF and extract text"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            print(f"Error loading PDF with PyPDFLoader: {e}")
            # Fallback: use pypdf directly if available
            try:
                from pypdf import PdfReader  # type: ignore
                reader = PdfReader(pdf_path)
                docs: List[Document] = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    docs.append(Document(page_content=text, metadata={"page": i}))
                print(f"Fallback loader extracted {len(docs)} pages")
                return docs
            except Exception as e2:
                print(f"Fallback PDF read failed: {e2}")
                return []
    
    def split_text(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        return split_docs
    
    def cluster_documents(self, documents: List[Document], n_clusters: int = 5) -> List[List[Document]]:
        """Cluster documents using embedding vectors + lightweight KMeans (NumPy)."""
        if not documents:
            return []
        if len(documents) < n_clusters:
            n_clusters = max(1, len(documents))

        texts = [doc.page_content for doc in documents]
        vectors = np.array(self.embeddings.embed_documents(texts), dtype=np.float32)

        # Normalize for cosine-like distance
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vectors_norm = vectors / norms

        labels = self._kmeans_numpy(vectors_norm, k=n_clusters, max_iter=50)

        clustered_docs = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            clustered_docs[int(label)].append(documents[idx])

        print(f"Clustered documents into {n_clusters} groups")
        return clustered_docs

    def _kmeans_numpy(self, X: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """Simple KMeans with NumPy using cosine distance via normalized vectors.
        Returns cluster labels for each row in X.
        """
        n = X.shape[0]
        rng = np.random.default_rng(42)
        # Initialize centroids by random unique choices
        init_idx = rng.choice(n, size=min(k, n), replace=False)
        centroids = X[init_idx].copy()

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            # Cosine similarity since vectors are normalized: argmax dot product
            sims = X @ centroids.T  # [n, k]
            new_labels = np.argmax(sims, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Recompute centroids
            for j in range(k):
                members = X[labels == j]
                if members.size == 0:
                    # Reinitialize empty cluster centroid randomly
                    centroids[j] = X[rng.integers(0, n)]
                else:
                    c = members.mean(axis=0)
                    norm = np.linalg.norm(c) + 1e-8
                    centroids[j] = c / norm

        return labels
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create and store vectors in ChromaDB"""
        if not documents:
            raise ValueError("No documents provided for vectorization")
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=Config.CHROMA_DB_PATH
        )
        vectorstore.persist()
        print(f"Created vectorstore with {len(documents)} documents")
        return vectorstore
    
    def extract_concepts(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract main concepts and relationships from documents"""
        # Combine all text
        combined_text = "\n".join([doc.page_content for doc in documents])
        
        # Prompt for concept extraction
        concept_prompt = """
        Analyze the following handwritten notes and extract:
        1. Main concepts/topics with clear definitions
        2. Key relationships between concepts (how they connect)
        3. Important definitions and key points
        4. Flashcards for learning the material
        
        Focus on creating a clear concept map structure with:
        - 5-10 main concepts that are central to the topic
        - Clear relationships showing how concepts connect
        - Definitions that explain each concept clearly
        - Flashcards that test understanding of key points
        
        Notes:
        {text}
        
        Return the analysis in JSON format with the following structure:
        {{
            "concepts": [
                {{
                    "name": "concept_name",
                    "definition": "clear and concise definition",
                    "importance": "high/medium/low"
                }}
            ],
            "relationships": [
                {{
                    "source": "concept1",
                    "target": "concept2",
                    "relationship": "clear relationship description"
                }}
            ],
            "flashcards": [
                {{
                    "question": "clear question about the concept",
                    "answer": "detailed answer explaining the concept",
                    "category": "category_name"
                }}
            ]
        }}
        """
        
        try:
            response = self.llm.invoke(concept_prompt.format(text=combined_text[:4000]))
            result = json.loads(response.content)
            
            # Ensure we have flashcards, if not generate them from concepts
            if not result.get("flashcards"):
                result["flashcards"] = self._generate_flashcards_from_concepts(result.get("concepts", []))
            
            return result
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            # Generate basic flashcards from document content as fallback
            fallback_flashcards = self._generate_fallback_flashcards(documents)
            return {"concepts": [], "relationships": [], "flashcards": fallback_flashcards}
    
    def _generate_flashcards_from_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate flashcards from extracted concepts"""
        flashcards = []
        categories = ["Key Concepts", "Definitions", "Important Terms"]
        
        for i, concept in enumerate(concepts):
            # Question about concept name
            flashcards.append({
                "question": f"What is the definition of '{concept['name']}'?",
                "answer": concept['definition'],
                "category": "Definitions"
            })
            
            # Question about importance
            flashcards.append({
                "question": f"What is the importance level of '{concept['name']}'?",
                "answer": f"{concept['name']} is of {concept['importance']} importance.",
                "category": "Key Concepts"
            })
        
        return flashcards
    
    def _generate_fallback_flashcards(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Generate basic flashcards from document content when concept extraction fails"""
        flashcards = []
        
        # Extract sentences that might make good flashcards
        for doc in documents:
            sentences = doc.page_content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                    # Create a simple Q&A from the sentence
                    words = sentence.split()
                    if len(words) > 5:
                        # Take first few words as question, rest as answer
                        question_words = words[:3]
                        answer_words = words[3:]
                        
                        if answer_words:
                            flashcards.append({
                                "question": f"What is {' '.join(question_words)}...?",
                                "answer": ' '.join(answer_words),
                                "category": "General"
                            })
        
        # Limit to 10 flashcards to avoid overwhelming
        return flashcards[:10]
    
    def build_concept_map(self, concepts_data: Dict[str, Any]) -> nx.DiGraph:
        """Build a directed graph representing the concept map"""
        G = nx.DiGraph()
        
        concepts = concepts_data.get("concepts", [])
        
        # Add nodes (concepts)
        for concept in concepts:
            G.add_node(concept["name"], 
                      definition=concept["definition"],
                      importance=concept["importance"])
        
        # Add edges (relationships)
        for rel in concepts_data.get("relationships", []):
            G.add_edge(rel["source"], rel["target"], 
                      relationship=rel["relationship"])
        
        # If no relationships were extracted, create basic connections
        if not concepts_data.get("relationships") and len(concepts) > 1:
            # Connect concepts in sequence or create a central hub
            for i in range(len(concepts) - 1):
                G.add_edge(concepts[i]["name"], concepts[i + 1]["name"], 
                          relationship="related to")
        
        return G
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        print("Starting PDF processing...")
        
        # Step 1: Load PDF
        documents = self.load_pdf(pdf_path)
        if not documents:
            return {"error": "Failed to load PDF"}
        
        # Step 2: Split text
        split_docs = self.split_text(documents)
        
        # Step 3: Cluster documents
        clustered_docs = self.cluster_documents(split_docs)
        
        # Step 4: Create vectorstore
        self.vectorstore = self.create_vectorstore(split_docs)
        
        # Step 5: Extract concepts
        concepts_data = self.extract_concepts(split_docs)
        
        # Store outputs for later API access
        self.documents = split_docs
        self.flashcards = concepts_data.get("flashcards", [])
        
        # Step 6: Build concept map
        self.concept_map = self.build_concept_map(concepts_data)
        
        # Prepare response
        response = {
            "concepts": concepts_data.get("concepts", []),
            "relationships": concepts_data.get("relationships", []),
            "flashcards": concepts_data.get("flashcards", []),
            "concept_map": self._graph_to_dict(self.concept_map),
            "clusters": len(clustered_docs),
            "total_chunks": len(split_docs)
        }
        
        print("PDF processing completed successfully!")
        return response
    
    def _graph_to_dict(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Convert networkx graph to dictionary for JSON serialization"""
        return {
            "nodes": [{"id": node, **G.nodes[node]} for node in G.nodes()],
            "edges": [{"source": u, "target": v, **G.edges[u, v]} 
                     for u, v in G.edges()]
        }
    
    def answer_question(self, question: str) -> str:
        """Answer questions using the RAG system"""
        if not self.vectorstore:
            return "No documents have been processed yet. Please upload a PDF first."
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        try:
            answer = qa_chain.run(question)
            return answer
        except Exception as e:
            return f"Error answering question: {e}"
