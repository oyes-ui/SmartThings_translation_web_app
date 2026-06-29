import json
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

local_gcp_json = "smartthings-explore-6d1922d106bd.json"
api_key = os.getenv("GEMINI_API_KEY")
if os.path.exists(local_gcp_json):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(local_gcp_json)
    with open(local_gcp_json, encoding="utf-8") as f:
        project_id = json.load(f).get("project_id", "smartthings-explore")
    client = genai.Client(vertexai=True, project=project_id, location=os.getenv("GCP_LOCATION", "global"))
elif api_key:
    client = genai.Client(api_key=api_key)
elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    client = genai.Client(vertexai=True, project=os.getenv("GCP_PROJECT_ID", "smartthings-explore"), location=os.getenv("GCP_LOCATION", "global"))
else:
    raise SystemExit("Gemini 인증 수단이 없습니다 (GEMINI_API_KEY 또는 서비스 계정 JSON 필요).")

for m in client.models.list():
    print(f"Name: {m.name}, Supported Actions: {m.supported_actions}")
