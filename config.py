import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your_secret_key_here')
    UPLOAD_FOLDER = 'uploads'
    CHROMA_DB_PATH = 'chroma_db'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
