import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. حدد مسار الملف الجديد اللي عايز تضيفه
new_file_path = r"D:\new\OneDrive\Desktop\Data_hac\YMOB_Consensus.pdf"
print(f"Loading new PDF from: {new_file_path}")
loader = PyPDFLoader(new_file_path)
raw_documents = loader.load()

# 2. تنظيف وتقطيع الملف الجديد
def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

for doc in raw_documents:
    doc.page_content = clean_text(doc.page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
new_chunks = text_splitter.split_documents(raw_documents)
print(f"Created {len(new_chunks)} new chunks.")

# 3. التحميل والربط بـ ChromaDB الحالية
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_folder = r"D:\new\OneDrive\Desktop\Data_hac\ccc"

vectorstore = Chroma(
    persist_directory=db_folder,
    embedding_function=embedding_model
)

# 4. إضافة الـ Chunks الجديدة فوق القديمة
print("Adding new documents to existing Vector DB...")
vectorstore.add_documents(new_chunks)

print("✅ New documents added successfully!")