# Garment Vision Service

Python 3.12 recommended (3.14 lacks some prebuilt packages).

    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    export GEMINI_API_KEY=your_key_here
    uvicorn vision_service:app --reload --host 0.0.0.0 --port 8000

API docs: http://localhost:8000/docs

Optional: `export GEMINI_MODEL=<model name>` to change the model.
