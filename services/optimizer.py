import itertools
from .backtester import backtest

def grid_optimize(df, tz_name, session_name="NY",
                  spans=(2,3), wicks=(1.3,1.5,1.8), rms=(1.5,1.8,2.0), volm=(1.0,1.2,1.5)):
    best=None
    for s, w, r, vm in itertools.product(spans,wicks,rms,volm):
        m = backtest(df, tz_name, session_name, span=s, wick_ratio=w, r_mult=r, vol_mult=vm)
        score = m["pf"]*m["winrate"]*max(1, m["trades"]/40)
        cand = {"span":s,"wick":w,"r":r,"volm":vm, **m, "score":round(score,3)}
        if not best or cand["score"]>best["score"]:
            best = cand
    return best
