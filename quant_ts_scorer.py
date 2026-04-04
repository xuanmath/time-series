import json

def calc_norm_score(raw_val, cfg):
    w = cfg["weight"]
    mode = cfg["mode"]
    if mode == "minimize":
        cap = cfg.get("soft_cap", 1.0)
        ratio = min(raw_val / cap, 1.0)
        norm = 1.0 - ratio
    else:
        floor = cfg.get("soft_floor", 0.0)
        clipped = max(raw_val, floor)
        norm = min(clipped, 1.0)
    return norm * w

def main():
    with open("project_metrics_cfg.json", encoding="utf-8") as f:
        meta_cfg = json.load(f)
    with open("run_export_metrics.json", encoding="utf-8") as f:
        run_data = json.load(f)

    total = 0.0
    for k, v_cfg in meta_cfg["metrics_weights"].items():
        total += calc_norm_score(run_data[k], v_cfg)

    final_score = round(total, 4)
    out = {
        "final_composite_score": final_score,
        "raw_metrics": run_data
    }
    with open("latest_baseline_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(final_score)

if __name__ == "__main__":
    main()