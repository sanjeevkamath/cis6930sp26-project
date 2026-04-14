import json
from pathlib import Path
from src.servers.load_server import insert_run_results

def load_all():
    results_dir = Path("results")
    
    # Grab every json file in results/
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in results/")
        return

    for file_path in json_files:
        print(f"Loading {file_path.name} into database...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            insert_run_results(data)
            
    print(f"\nSuccessfully loaded {len(json_files)} runs into data/travel_advisory.db!")

if __name__ == "__main__":
    load_all()
