from __future__ import annotations

import pandas as pd

from etf_mapper.spade_checks import check_option_chain, check_price_history


def test_chain_empty_fails():
    checks = check_option_chain(pd.DataFrame(), pd.DataFrame())
    assert any(c.code == "CHAIN_EMPTY" and c.level == "FAIL" for c in checks)


def test_bid_gt_ask_warns():
    calls = pd.DataFrame([
        {"strike": 100, "bid": 2.0, "ask": 1.0},
        {"strike": 101, "bid": 1.0, "ask": 1.1},
    ])
    puts = pd.DataFrame([
        {"strike": 100, "bid": 1.0, "ask": 1.2},
    ])
    checks = check_option_chain(calls, puts)
    assert any(c.code == "BID_GT_ASK" for c in checks)


def test_history_empty_fails():
    checks = check_price_history(pd.DataFrame())
    assert any(c.code == "HISTORY_EMPTY" and c.level == "FAIL" for c in checks)
