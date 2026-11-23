import pytest
from src.common import ProtocolError
from src.compute_metrics import metrics
from src.annotate import agreement, validate_ratings


def test_metrics():
    result = metrics(["P", "P", "S", "N"], [True, False, True])
    assert result["H"] == result["H_star"] == .5
    assert result["C"] == 2/3
    assert result["O"] == 1/3 or result["O"] == 1-2/3


@pytest.mark.parametrize("kind", ["empty", "refusal", "no_clinical_claims"])
def test_empty_denominator(kind):
    result = metrics([], [False], kind)
    assert result["H"] is None and result["H_star"] == 1 and result["C"] == 0


def test_zero_required_facts_blocked():
    with pytest.raises(ProtocolError):
        metrics(["P"], [])


def test_agreement_before_consensus():
    a = [{"claim_id": str(i), "label": l} for i, l in enumerate(["P", "S", "N", "P"])]
    b = [{"claim_id": str(i), "label": l} for i, l in enumerate(["P", "N", "N", "P"])]
    result = agreement(a, b)
    assert result["raw_agreement"] == .75
    assert result["cohen_kappa"] == pytest.approx(.6)
    assert result["disagreement_matrix"][1][2] == 1


def test_coverage_needs_supported_claim():
    rating = {"labels": [{"claim_id": "c1", "label": "N", "rationale": "Not in transcript"}],
              "coverage": [{"fact_id": "f1", "covered": True, "claim_ids": ["c1"], "rationale": "Claim present"}]}
    with pytest.raises(ProtocolError, match="supported"):
        validate_ratings(rating, [{"claim_id": "c1"}], [{"fact_id": "f1"}], "source")
