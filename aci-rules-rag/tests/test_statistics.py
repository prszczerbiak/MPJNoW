import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests
from scipy.stats import permutation_test, ttest_1samp
from src.aggregate_results import aggregate, paired_bootstrap, paired_tests
from src.common import ProtocolError


def example_means():
    return pd.DataFrame([{"encounter_id": str(i), "variant": v,
                          "H_star": .3 + .02*i - (.08 if v == "C" else 0),
                          "C": .6 + .01*i + (.01*i if v == "C" else 0)}
                         for i in range(8) for v in "EABC"])


def test_prespecified_tests_and_holm():
    means = example_means()
    result = paired_tests(means, n_resamples=999, bootstrap_resamples=200)
    tests = result["tests"]
    assert len(tests) == 4
    expected = multipletests([t["p_raw"] for t in tests], method="holm")[1]
    assert np.allclose([t["p_holm"] for t in tests], expected)
    a = means[means.variant == "A"].H_star.to_numpy()
    c = means[means.variant == "C"].H_star.to_numpy()
    reference = permutation_test((c, a), lambda x, y, axis: np.mean(x-y, axis=axis),
                                 permutation_type="samples", alternative="less", n_resamples=999,
                                 rng=np.random.default_rng(20261021))
    assert tests[0]["p_raw"] == reference.pvalue
    diff = means[means.variant == "C"].C.to_numpy() - means[means.variant == "A"].C.to_numpy()
    assert tests[2]["p_raw"] == ttest_1samp(diff, -.05, alternative="greater").pvalue
    assert all(t["comparison"] in ("C-A", "C-B") for t in tests)


def test_bootstrap_reproducible():
    assert paired_bootstrap([0, 1, 2], 200) == paired_bootstrap([0, 1, 2], 200)
    assert paired_bootstrap([2, 2, 2], 200) == [2, 2]


def synthetic_frame():
    return pd.DataFrame([{"task_id": f"{i}-{v}-{s}", "run_id": "test", "environment_id": "cpu-test",
                          "encounter_id": str(i), "variant": v, "seed": s, "split": "test1", "status": "ok",
                          "H": float(i), "H_star": float(i), "C": .5, "O": .5}
                         for i in range(2) for v in "EABC" for s in ([None] if v == "E" else [17, 42, 2026])])


def test_two_level_macro_average():
    frame = synthetic_frame()
    means, macro, audit = aggregate(frame)
    assert len(means) == 8
    assert (macro.H_star == .5).all()
    assert (macro.n_consultations == 2).all()
    assert audit["technical_missing_fraction"]["C"] == 0


def test_missingness_and_environment_guards():
    frame = synthetic_frame()
    frame.loc[frame.variant == "C", "status"] = "technical_failure"
    with pytest.raises(ProtocolError, match="5%"):
        aggregate(frame)
    frame = synthetic_frame()
    frame.loc[0, "environment_id"] = "other"
    with pytest.raises(ProtocolError, match="environments"):
        aggregate(frame)


def test_unexpected_seed_rejected():
    frame = synthetic_frame()
    frame.loc[frame.seed == 17, "seed"] = 18
    with pytest.raises(ProtocolError, match="repetitions"):
        aggregate(frame)


def test_seed_means_then_consultation_means():
    frame = synthetic_frame()
    mask = (frame.encounter_id == "0") & (frame.variant == "A")
    frame.loc[mask, "H_star"] = [0, 0, .9]
    frame.loc[(frame.encounter_id == "1") & (frame.variant == "A"), "H_star"] = .9
    means, macro, _ = aggregate(frame)
    assert means[(means.encounter_id == "0") & (means.variant == "A")].H_star.item() == pytest.approx(.3)
    assert macro[macro.variant == "A"].H_star.item() == pytest.approx(.6)


def test_small_technical_missingness_does_not_average_two_seeds():
    base = synthetic_frame()
    frames = []
    for i in range(10):
        part = base.copy()
        part.encounter_id = part.encounter_id.map(lambda eid: str(2*i+int(eid)))
        part.task_id = part.task_id.map(lambda tid: f"{i}-{tid}")
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    mask = (frame.encounter_id == "0") & (frame.variant == "C") & (frame.seed == 17)
    frame.loc[mask, "status"] = "technical_failure"
    means, macro, audit = aggregate(frame)
    assert not ((means.encounter_id == "0") & (means.variant == "C")).any()
    assert audit["technical_missing_fraction"]["C"] == pytest.approx(1/60)
    assert macro[macro.variant == "C"].n_consultations.item() == 19


def test_scheduled_task_accounting():
    frame = synthetic_frame()
    with pytest.raises(ProtocolError, match="every scheduled"):
        aggregate(frame, [{"task_id": "missing"}])
