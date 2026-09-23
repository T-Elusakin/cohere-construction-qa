import os

# src/store.py, src/retrieval.py, and src/generate.py each read
# COHERE_API_KEY at import time to construct a Cohere client. Tests mock
# every actual API call, so a real key is never needed - but a value must
# exist in the environment for those imports to succeed at all (e.g. in CI,
# or a fresh clone with no .env file).
os.environ.setdefault("COHERE_API_KEY", "test-key-not-used")
