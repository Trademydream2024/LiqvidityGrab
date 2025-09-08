import sqlite3, json, os

DB="results.db"

def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS optim_results(
            symbol TEXT, session TEXT, params_json TEXT, metrics_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

def save_result(symbol, session, params, metrics):
    with sqlite3.connect(DB) as c:
        c.execute("INSERT INTO optim_results(symbol,session,params_json,metrics_json) VALUES (?,?,?,?)",
                  (symbol, session, json.dumps(params), json.dumps(metrics)))

def fetch_latest(symbol, session):
    with sqlite3.connect(DB) as c:
        cur=c.execute("""SELECT params_json, metrics_json, created_at
                         FROM optim_results
                         WHERE symbol=? AND session=?
                         ORDER BY created_at DESC LIMIT 1""",(symbol,session))
        return cur.fetchone()
