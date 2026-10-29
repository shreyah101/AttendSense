import pytest
from src.attendance import percentage, plan, projected, safe_bunks_now, meets_target, certain_risk


def test_percentage_and_empty():
    assert percentage(35, 43) == pytest.approx(81.3953488372)
    assert percentage(0, 0) is None
    assert meets_target(0, 0) is None
    assert plan(0, 0, 0).maximum is None


@pytest.mark.parametrize("a,c,r,t,required,allowance", [
    (3,4,4,75,3,1), (35,43,10,75,5,5), (3,4,0,75,0,0),
    (0,4,2,75,5,None), (0,0,4,75,3,1), (4,4,2,100,2,0),
    (0,0,100,1,1,99), (29,100,0,29,0,0)])
def test_plan(a,c,r,t,required,allowance):
    result = plan(a,c,r,t)
    assert result.required == required
    assert result.allowance == allowance
    assert result.feasible == (allowance is not None)


@pytest.mark.parametrize("a,c,t,bunks", [(3,4,75,0),(4,4,75,1),(35,43,75,3),(0,0,75,0),(2,4,75,0),(10,10,100,0),(29,100,29,0)])
def test_bunks(a,c,t,bunks):
    assert safe_bunks_now(a,c,t) == bunks
    if meets_target(a,c,t):
        assert meets_target(a,c+bunks,t)
        assert not meets_target(a,c+bunks+1,t)


def test_outcomes():
    assert projected(3,4,4,3) == 75
    assert certain_risk(4,4,1) == 0
    assert certain_risk(0,4,1) == 1
    assert certain_risk(3,4,0) == 0
    assert certain_risk(2,4,0) == 1
    assert certain_risk(3,4,4) is None
    assert certain_risk(0,0,0) is None


@pytest.mark.parametrize("args", [(-1,2,0,75),(3,2,0,75),(1,2,-1,75),(1.0,2,0,75),(True,2,0,75),(1,2,0,0),(1,2,0,101),(1,2,0,float('nan'))])
def test_invalid(args):
    with pytest.raises(ValueError):
        plan(*args)


def test_invalid_scenario():
    for x in [-1,3,True,1.5]:
        with pytest.raises(ValueError):
            projected(1,2,2,x)


def test_exhaustive_small_counts():
    # Independent enumeration checks the closed-form allowance at equality boundaries.
    for c in range(1,12):
        for a in range(c+1):
            for r in range(6):
                for threshold in [1,29,75,100]:
                    valid = [x for x in range(r+1) if 100*(a+x) >= threshold*(c+r)]
                    p = plan(a,c,r,threshold)
                    assert p.feasible == bool(valid)
                    assert p.allowance == (r-min(valid) if valid else None)
