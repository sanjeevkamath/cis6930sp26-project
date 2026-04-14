import json
from src.servers.load_server import list_snapshots
from src.servers.stability_analysis import compare_conditions

def run_analysis():
    # 1. Grab all the metadata for our loaded runs
    snapshots = list_snapshots()
    
    # 2. Sort the runs into our 2x2 Experimental Matrix based on model + condition
    llama_constrained = [s["run_id"] for s in snapshots if "llama" in str(s["model"]).lower() and s["constrained"] == 1]
    llama_unconstrained = [s["run_id"] for s in snapshots if "llama" in str(s["model"]).lower() and s["constrained"] == 0]
    
    gpt_constrained = [s["run_id"] for s in snapshots if ("gpt" in str(s["model"]).lower() or "120b" in str(s["model"]).lower()) and s["constrained"] == 1]
    gpt_unconstrained = [s["run_id"] for s in snapshots if ("gpt" in str(s["model"]).lower() or "120b" in str(s["model"]).lower()) and s["constrained"] == 0]

    # 3. Compute Stability Analysis for Llama 8B
    print("========================================")
    print("      SLM: LLAMA-3.1-8B-INSTRUCT        ")
    print("========================================")
    if len(llama_constrained) >= 2 and len(llama_unconstrained) >= 2:
        print(f"Comparing {len(llama_constrained)} constrained vs {len(llama_unconstrained)} unconstrained runs...\n")
        res_llama = compare_conditions(llama_constrained, llama_unconstrained)
        print("Constrained CI: ", res_llama["constrained"]["primary"]["row_jaccard"])
        print("Unconstrained CI:", res_llama["unconstrained"]["primary"]["row_jaccard"])
        print(json.dumps(res_llama["volatility_reduction"], indent=2))
    else:
        print(f"Error: Need at least 2 runs. Found {len(llama_constrained)} constrained, {len(llama_unconstrained)} unconstrained.")

    print("\n========================================")
    print("         LLM: GPT-OSS-120B              ")
    print("========================================")
    if len(gpt_constrained) >= 2 and len(gpt_unconstrained) >= 2:
        print(f"Comparing {len(gpt_constrained)} constrained vs {len(gpt_unconstrained)} unconstrained runs...\n")
        res_gpt = compare_conditions(gpt_constrained, gpt_unconstrained)
        print("Constrained CI: ", res_gpt["constrained"]["primary"]["row_jaccard"])
        print("Unconstrained CI:", res_gpt["unconstrained"]["primary"]["row_jaccard"])
        print(json.dumps(res_gpt["volatility_reduction"], indent=2))
    else:
        print(f"Error: Need at least 2 runs. Found {len(gpt_constrained)} constrained, {len(gpt_unconstrained)} unconstrained.")


if __name__ == "__main__":
    run_analysis()
