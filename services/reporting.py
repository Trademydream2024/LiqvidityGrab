def render_session_stats(symbol, stats: dict):
    s = [f"📈 <b>{symbol}</b> — Session Stats (last 12m)"]
    for name, m in stats.items():
        s.append(f"• {name}: bars={m['bars']}, avg range={m['avg_range_pct']}%, p95={m['p95_range_pct']}%")
    return "\n".join(s)

def render_bt(symbol, sess, m, params=None):
    base = (
        f"📊 <b>{symbol}</b> — Backtest [{sess}]\n"
        f"• Trades: <b>{m['trades']}</b>\n"
        f"• Winrate: <b>{m['winrate']*100:.1f}%</b>\n"
        f"• PF: <b>{m['pf']}</b>\n"
        f"• MaxDD (R): <b>{m['max_dd']}</b>"
    )
    if params:
        base += f"\n🧩 Params: span={params['span']}, wick≥{params['wick']}, R={params['r']}, volx≥{params['volm']}"
    return base
