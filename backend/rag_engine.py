from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = Path("../data/faiss_index")
MANUALS_DIR = Path("../data/manuals")

class TwinRepRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vectorstore = self._load_or_create_index()

    def _load_or_create_index(self) -> FAISS:
        if (INDEX_DIR / "index.faiss").exists():
            return FAISS.load_local(str(INDEX_DIR), self.embeddings, allow_dangerous_deserialization=True)
        # Initialize with seed document so FAISS is immediately queryable
        vs = FAISS.from_texts(
            texts=["TwinRep AI Technical After-Sales Knowledge Base initialized."],
            embedding=self.embeddings,
            metadatas=[{"source": "system_init"}],
        )
        vs.save_local(str(INDEX_DIR))
        return vs

    def _extract_text(self, file_path: Path) -> str:
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def ingest_file(self, file_path: Path) -> int:
        raw_text = self._extract_text(file_path).strip()
        if not raw_text:
            return 0
        chunks = self.splitter.split_text(raw_text)
        metadatas = [{"source": file_path.name} for _ in chunks]
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        self.vectorstore.save_local(str(INDEX_DIR))
        return len(chunks)

    def ingest_existing_manuals(self) -> int:
        total_chunks = 0
        for file_path in MANUALS_DIR.glob("*"):
            if file_path.suffix.lower() in {".pdf", ".txt", ".md"}:
                total_chunks += self.ingest_file(file_path)
        return total_chunks

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        docs = self.vectorstore.similarity_search(query, k=k)
        return [
            {"source": d.metadata.get("source", "manual"), "content": d.page_content}
            for d in docs
            if d.metadata.get("source") != "system_init"
        ]

    def add_verified_resolution(self, ticket_id: int, question: str, verified_answer: str):
        """Feature 3: Hot-updates the live FAISS index when a manager approves or edits a reply."""
        entry = (
            f"Verified Support Resolution (Ticket #{ticket_id})\n"
            f"Customer Question: {question}\n"
            f"Approved Technical Answer: {verified_answer}"
        )
        chunks = self.splitter.split_text(entry)
        metadatas = [{"source": f"verified_ticket_{ticket_id}"} for _ in chunks]
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        self.vectorstore.save_local(str(INDEX_DIR))
