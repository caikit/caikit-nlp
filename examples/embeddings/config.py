# Standard
import os

port = (
    os.getenv("CAIKIT_EMBEDDINGS_PORT") if os.getenv("CAIKIT_EMBEDDINGS_PORT") else 8085
)
host = (
    os.getenv("CAIKIT_EMBEDDINGS_HOST")
    if os.getenv("CAIKIT_EMBEDDINGS_HOST")
    else "localhost"
)
