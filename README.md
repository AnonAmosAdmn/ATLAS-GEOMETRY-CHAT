pip install -r requirements.txt

python -m nltk.downloader wordnet omw-1.4

--------------------------------------------

Ollama must be installed separately, then:

--------------------------------------------

ollama pull qwen2.5:7b

ollama serve

--------------------------------------------

streamlit run app.py
