from dotenv import load_dotenv
import os

load_dotenv()  # reads .env from the current folder

# sanity check — don't print the full key, just confirm it's loaded
key = os.getenv("LANGCHAIN_API_KEY")
print("Key loaded:", bool(key), "| starts with:", key[:8] if key else None)

from langsmith import Client
client = Client()
projects = client.list_projects()
for p in projects:
    print(p.name)

print("\n--- Checking for actual runs ---")
for name in ["my-local-rag", "default"]:
    runs = list(client.list_runs(project_name=name, limit=5))
    print(f"{name}: {len(runs)} recent runs found")