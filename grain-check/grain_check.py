"""Grain Check: execution-based evaluation of retail analytics SQL. Python 3.9+."""
import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

# Each sale is (product, units sold, units returned, unit price in cents).
FIXTURES = {
    "launch": [("Atlas", 100, 8, 20), ("Orbit", 10, 4, 150), ("Atlas", 2, 0, 20)],
    "shift": [("Orbit", 40, 8, 30), ("Atlas", 4, 2, 100), ("Orbit", 5, 0, 30)],
    "uneven": [("Kite", 1, 1, 90), ("Kite", 99, 0, 90), ("Flux", 20, 4, 400)],
}

CASES = {
    "return_rate": {
        "question": "Which product has the highest returned units / sold units? Break ties by name.",
        "reference": "SELECT product FROM sales GROUP BY product ORDER BY 1.0 * SUM(returned) / SUM(units) DESC, product LIMIT 1",
        "baseline": "SELECT product FROM sales GROUP BY product ORDER BY SUM(returned) DESC, product LIMIT 1",
    },
    "net_revenue": {
        "question": "What is net revenue in cents after subtracting returned units at the original unit price?",
        "reference": "SELECT SUM((units - returned) * price_cents) FROM sales",
        "baseline": "SELECT SUM(units * price_cents) FROM sales",
    },
    "repeat_products": {
        "question": "Which products occur on two or more sales rows? Return product names only.",
        "reference": "SELECT product FROM sales GROUP BY product HAVING COUNT(*) >= 2",
        "baseline": "SELECT DISTINCT product FROM sales WHERE units >= 2",
    },
}


def expected(case, rows):
    totals = {}
    revenue = 0
    for product, units, returned, price in rows:
        group = totals.setdefault(product, [0, 0, 0])
        group[0] += units
        group[1] += returned
        group[2] += 1
        revenue += (units - returned) * price
    if case == "net_revenue":
        return [(revenue,)]
    if case == "repeat_products":
        return [(p,) for p, counts in totals.items() if counts[2] >= 2]
    winner = min(totals, key=lambda p: (-totals[p][1] / totals[p][0], p))
    return [(winner,)]


def evaluate(case, rows, sql):
    target = expected(case, rows)
    result = {"expected": target, "passed": False}
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE TABLE sales(product TEXT, units INTEGER, returned INTEGER, price_cents INTEGER)")
        db.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", rows)
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        db.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        db.set_progress_handler(lambda: 1, 100000)
        try:
            actual = db.execute(sql).fetchmany(101)
            result.update(actual=actual, passed=Counter(actual) == Counter(target))
        except sqlite3.Error as error:
            result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["baseline", "reference"], default="baseline")
    parser.add_argument("--candidates", type=Path, help="JSON object mapping all three case IDs to SQL strings")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-pass", action="store_true", help="Exit 1 on a mismatch")
    args = parser.parse_args()
    queries = {k: v[args.suite] for k, v in CASES.items()}
    if args.candidates:
        try:
            queries = json.loads(args.candidates.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if not isinstance(queries, dict) or queries.keys() != CASES.keys() or not all(isinstance(q, str) for q in queries.values()):
            parser.error("Supply one SQL string for each of: " + ", ".join(CASES))
    report = [{"case": case, "fixture": name, "question": CASES[case]["question"],
               "sql": sql, "rows": rows, **evaluate(case, rows, sql)}
              for case, sql in queries.items() for name, rows in FIXTURES.items()]
    passed = sum(row["passed"] for row in report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for row in report:
            print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['case']:16} {row['fixture']:8} expected={row['expected']} got={row.get('actual', row.get('error'))}")
        print(f"{passed}/{len(report)} fixture checks passed")
    return int(args.require_pass and passed != len(report))


if __name__ == "__main__":
    raise SystemExit(main())
