"""Zaplanowana analiza sparowana na test1. Wariant E służy wyłącznie do opisu."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import permutation_test, ttest_1samp
from statsmodels.stats.multitest import multipletests

from .common import ProtocolError, atomic_write, read_json, write_json

SEEDS = {17, 42, 2026}
MEASURES = ["H", "H_star", "C", "O"]


def aggregate(frame, expected_tasks=None):
    if frame.empty or set(frame["split"]) != {"test1"}:
        raise ProtocolError("Primary analysis requires nonempty test1 data only")
    if frame["run_id"].nunique(dropna=False) != 1 or frame["environment_id"].nunique(dropna=False) != 1:
        raise ProtocolError("Never pool runs or computational environments")
    if "evaluation_method" in frame and (frame.evaluation_method.isna().any() or set(frame.evaluation_method) != {"human"}):
        raise ProtocolError("Only human assessments are supported")
    if frame.duplicated(["encounter_id", "variant", "seed"]).any() or frame["task_id"].duplicated().any():
        raise ProtocolError("Duplicate task or repetition")
    if expected_tasks is not None and set(frame["task_id"]) != {t["task_id"] for t in expected_tasks}:
        raise ProtocolError("Metrics must account for every scheduled task, including technical failures")
    if not set(frame["status"]) <= {"ok", "technical_failure"}:
        raise ProtocolError("Unknown task status")
    encounters = set(frame["encounter_id"])
    missing_rates = {}
    for variant in "EABC":
        sub = frame[frame.variant == variant]
        if set(sub.encounter_id) != encounters:
            raise ProtocolError("All variants must retain the same planned encounters")
        for _, group in sub.groupby("encounter_id"):
            if variant == "E":
                valid = len(group) == 1 and group.seed.isna().all()
            else:
                valid = len(group) == 3 and set(group.seed) == SEEDS
            if not valid:
                raise ProtocolError("Missing or unexpected scheduled repetitions")
        missing_rates[variant] = float((sub.status != "ok").mean())
    if any(rate > 0.05 for rate in missing_rates.values()):
        raise ProtocolError(f"Technical missingness exceeds 5%; primary inference stopped: {missing_rates}")
    ok = frame[frame.status == "ok"]
    if ok[["H_star", "C", "O"]].isna().any().any():
        raise ProtocolError("Successful responses require completed assessments")
    values = ok[MEASURES].to_numpy(dtype=float)
    if np.isinf(values).any() or np.any((values < 0) | (values > 1)):
        raise ProtocolError("Metrics outside [0,1]")
    # Braki techniczne pozostają w danych. Średnia wymaga kompletu trzech powtórzeń.
    complete = []
    incomplete = []
    for (eid, variant), group in frame.groupby(["encounter_id", "variant"]):
        if (group.status == "ok").all():
            complete.append({"encounter_id": eid, "variant": variant,
                             "H_defined_repetitions": int(group.H.notna().sum()),
                             **{m: float(group[m].mean()) if group[m].notna().any() else None for m in MEASURES}})
        else:
            incomplete.append({"encounter_id": eid, "variant": variant})
    means = pd.DataFrame(complete)
    macro = means.groupby("variant")[MEASURES].mean().reset_index()
    macro["n_consultations"] = macro.variant.map(means.groupby("variant").size())
    macro["H_defined_consultations"] = macro.variant.map(means.groupby("variant").H.count())
    return means, macro, {"technical_missing_fraction": missing_rates, "incomplete_repetition_groups": incomplete}


def paired_bootstrap(differences, n_resamples=10000, seed=20261021):
    x = np.asarray(differences, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ProtocolError("Bootstrap requires at least two finite consultation differences")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 1000):
        size = min(1000, n_resamples-start)
        means[start:start+size] = x[rng.integers(0, len(x), size=(size, len(x)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def paired_tests(means, n_resamples=99999, bootstrap_resamples=10000, seed=20261021):
    # Wszystkie cztery testy obejmują te same kompletne konsultacje A/B/C, bez wariantu E.
    primary = means[means.variant.isin(list("ABC"))]
    h = primary.pivot(index="encounter_id", columns="variant", values="H_star").reindex(columns=list("ABC"))
    c = primary.pivot(index="encounter_id", columns="variant", values="C").reindex(columns=list("ABC"))
    keep = h.notna().all(axis=1) & c.notna().all(axis=1)
    excluded = h.index[~keep].tolist()
    h, c = h.loc[keep], c.loc[keep]
    if len(h) < 2:
        raise ProtocolError("Fewer than two complete paired consultations")
    results = []
    for metric, table in (("H_star", h), ("C", c)):
        for comparator in ("A", "B"):
            differences = (table["C"]-table[comparator]).to_numpy(dtype=float)
            if metric == "H_star":
                result = permutation_test((table["C"].to_numpy(), table[comparator].to_numpy()),
                                          lambda x, y, axis: np.mean(x-y, axis=axis),
                                          permutation_type="samples", alternative="less", vectorized=True,
                                          n_resamples=n_resamples, batch=1000, rng=np.random.default_rng(seed))
                pvalue, statistic = float(result.pvalue), float(result.statistic)
                test, null = "paired_permutation_less", 0.0
            else:
                # Zerowa wariancja różnic uniemożliwia ten test; protokół nie dopuszcza zastąpienia go innym.
                if np.std(differences, ddof=1) == 0:
                    raise ProtocolError("Degenerate paired t-test: zero variance; no confirmatory claim is produced")
                result = ttest_1samp(differences, popmean=-0.05, alternative="greater")
                pvalue, statistic = float(result.pvalue), float(result.statistic)
                test, null = "one_sample_paired_difference_t_greater", -0.05
            if not np.isfinite(pvalue):
                raise ProtocolError("Nonfinite pre-specified test p-value")
            results.append({"comparison": f"C-{comparator}", "metric": metric, "test": test,
                            "null_value": null, "mean_difference": float(differences.mean()),
                            "statistic": statistic, "p_raw": pvalue, "n_consultations": len(h),
                            "bootstrap_percentile_95_ci": paired_bootstrap(differences, bootstrap_resamples, seed)})
    reject, adjusted, _, _ = multipletests([r["p_raw"] for r in results], alpha=0.05, method="holm")
    for row, p, rejected in zip(results, adjusted, reject):
        row.update(p_holm=float(p), reject_holm=bool(rejected))
    return {"tests": results, "paired_encounters": h.index.tolist(), "unpaired_encounters": excluded,
            "permutation_resamples": n_resamples, "bootstrap_resamples": bootstrap_resamples,
            "resampling_seed": seed, "alpha": 0.05, "bootstrap_unit": "consultation",
            "bootstrap_interval": "two-sided percentile 95%, descriptive and not multiplicity-adjusted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/metrics.csv")
    parser.add_argument("--plan", default="outputs/frozen_plan.json")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    plan = read_json(args.plan)
    if plan["blocked"]:
        raise ProtocolError("Primary inference blocked by prompt exclusions")
    frame = pd.read_csv(args.metrics)
    expected_method = "human"
    methods = set(frame.evaluation_method) if "evaluation_method" in frame else {"human"}
    if methods != {expected_method}:
        raise ProtocolError("Explicit analysis mode must match assessment provenance")
    means, macro, audit = aggregate(frame, plan["tasks"])
    from pathlib import Path
    output = Path(args.output)
    means["evaluation_method"] = expected_method
    macro["evaluation_method"] = expected_method
    audit["evaluation_method"] = expected_method
    write_json(output / "missingness.json", audit)
    atomic_write(output / "consultation_means.csv", means.to_csv(index=False))
    atomic_write(output / "macro_means.csv", macro.to_csv(index=False))
    statistics = paired_tests(means)
    statistics.update(evaluation_method=expected_method, inference_scope="human_assessed")
    write_json(output / "statistics.json", statistics)


if __name__ == "__main__":
    main()
