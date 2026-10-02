# FRIDAE V2 Web

Frontend: Streamlit.

Environment:
- `FRIDAE_API_URL=https://f-r-i-d-a-e.onrender.com`
- `FRIDAE_API_TOKEN=<backend EVAL_TOKEN>`

Expected backend routes:
- `POST /api/chat`
- `POST /api/upload`
- `GET /api/dataset`
- `GET /api/artifacts`
- `GET /api/artifacts/{artifact_id}`
- existing `GET /run.jsonl`

Render direct deployment:
- Root: `frontend`
- Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`

Docker deployment:
- Use `frontend.Dockerfile`
