import requests
import json

query = """
query Model($entity: String!, $project: String!, $run: String!) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      id
      name
      state
      summaryMetrics
      createdAt
      heartbeatAt
    }
  }
}
"""
variables = {
    "entity": "bulyakovbr-bogdan-buliakov",
    "project": "BEBLaDII-Phase1-PlanC",
    "run": "tql5y6vf"
}
resp = requests.post("https://api.wandb.ai/graphql", json={"query": query, "variables": variables})
print("Status:", resp.status_code)
try:
    data = resp.json()
    run_data = data.get("data", {}).get("project", {}).get("run", {})
    print("Run data:", run_data.get("name"), run_data.get("state"))
    if run_data.get("summaryMetrics"):
        metrics = json.loads(run_data["summaryMetrics"])
        print("Filtered metrics:")
        for k in sorted(metrics.keys()):
            if not k.startswith("_"):
                print(f"  {k}: {metrics[k]}")
except Exception as e:
    print("Error:", e, resp.text[:500])
