from os import environ

# Production Settings
DEBUG = False
ALLOWED_HOSTS = ['*']  # Update this with your actual domain

# API Configuration
OPENAI_API_KEY = environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = environ.get('PINECONE_API_KEY')

# Server Settings
HOST = '0.0.0.0'
PORT = int(environ.get('PORT', 8000))

# Model Settings
MODEL_DEVICE = 'cpu'  # Change to 'cuda' if GPU is available
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4-turbo-preview"

# Cache Settings
DOCUMENT_CACHE_SIZE = 1000
REDIS_URL = environ.get('REDIS_URL')

# Security Settings
CORS_ORIGINS = environ.get('CORS_ORIGINS', '').split(',')
SSL_CERT = environ.get('SSL_CERT')
SSL_KEY = environ.get('SSL_KEY') 