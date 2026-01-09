"""Public-safe configuration module.

Do NOT commit real credentials to source control.
Populate these via environment variables (or a .env file loaded in your runtime).
"""

import os

def _get(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None else default

# ---- Zilliz / Milvus ----
ZILLIZ_URI = _get("ZILLIZ_URI", "https://YOUR-ZILLIZ-ENDPOINT")  # e.g., https://xxx.api.gcp-us-west1.zillizcloud.com
ZILLIZ_API_KEY = _get("ZILLIZ_API_KEY", "YOUR_ZILLIZ_API_KEY")
ZILLIZ_DB_NAME = _get("ZILLIZ_DB_NAME", "default")

# ---- SQL / Dynamics sync (optional) ----
PappsDB_Server = _get("PAPPSDB_SERVER", "YOUR_SQL_SERVER_HOST")
PappsDB_UserID = _get("PAPPSDB_USER", "YOUR_SQL_USERNAME")
PappsDB_PW = _get("PAPPSDB_PASSWORD", "YOUR_SQL_PASSWORD")

# ---- Azure OpenAI (used by API_Azure wrapper) ----
AZURE_OPENAI_ENDPOINT = _get("AZURE_OPENAI_ENDPOINT", "https://YOUR-AOAI-RESOURCE.openai.azure.com")
AZURE_OPENAI_API_KEY = _get("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = _get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = _get("AZURE_OPENAI_CHAT_DEPLOYMENT", "YOUR_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBED_DEPLOYMENT = _get("AZURE_OPENAI_EMBED_DEPLOYMENT", "YOUR_EMBED_DEPLOYMENT")

# ---- Azure Document Intelligence (if used by API_Azure / OCR wrapper) ----
AZURE_DOCINTEL_ENDPOINT = _get("AZURE_DOCINTEL_ENDPOINT", "https://YOUR-DOCINTEL.cognitiveservices.azure.com")
AZURE_DOCINTEL_KEY = _get("AZURE_DOCINTEL_KEY", "YOUR_DOCINTEL_KEY")
