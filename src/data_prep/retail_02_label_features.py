"""Online Retail II: monthly slices with cut-off date C (end of month), March 2010 - August 2011.
Population of a slice: customers with >= 1 purchase in the 365 days before C.
Churn: no purchase in (C, C + 90 days].
Features use only transactions dated <= C.

Usage: python retail_02_label_features.py <work_dir containing retail.parquet> <output_dir>
Output: <output_dir>/retail_monthly.parquet
"""
import duckdb, sys
KK, OUT = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute(f"""CREATE VIEW t AS SELECT Invoice, StockCode, Quantity, InvoiceDate, Price, CAST("Customer ID" AS INTEGER) cid, Country,
   starts_with(Invoice, 'C') AS is_return, Quantity * Price AS amount
   FROM '{KK}/retail.parquet' WHERE "Customer ID" IS NOT NULL""")
con.execute("CREATE VIEW p AS SELECT * FROM t WHERE NOT is_return AND Quantity > 0 AND Price > 0")
cut = con.execute("SELECT DISTINCT last_day(d)::DATE c FROM (SELECT unnest(generate_series(DATE '2010-03-01', DATE '2011-08-01', INTERVAL 1 MONTH)) d) ORDER BY c").fetchall()
parts = []
for (C,) in cut:
    parts.append(f"""
    SELECT DATE '{C}' AS cutoff, h.cid,
      CASE WHEN f.cid IS NULL THEN 1 ELSE 0 END AS is_churn,
      date_diff('day', h.last_buy, DATE '{C}') AS recency_days,
      date_diff('day', h.first_buy, DATE '{C}') AS tenure_days,
      h.n_inv_365, h.n_inv_90, h.amount_365, h.amount_90, h.amount_365 / h.n_inv_365 AS avg_basket,
      h.n_products_365, h.n_active_months_365,
      coalesce(r.n_returns_365, 0) AS n_returns_365, coalesce(r.return_amount_365, 0) AS return_amount_365,
      h.uk
    FROM (SELECT cid, max(InvoiceDate) last_buy, min(InvoiceDate) first_buy,
            count(DISTINCT Invoice) FILTER (WHERE InvoiceDate > DATE '{C}' - INTERVAL 365 DAY) n_inv_365,
            count(DISTINCT Invoice) FILTER (WHERE InvoiceDate > DATE '{C}' - INTERVAL 90 DAY) n_inv_90,
            sum(amount) FILTER (WHERE InvoiceDate > DATE '{C}' - INTERVAL 365 DAY) amount_365,
            coalesce(sum(amount) FILTER (WHERE InvoiceDate > DATE '{C}' - INTERVAL 90 DAY), 0) amount_90,
            count(DISTINCT StockCode) FILTER (WHERE InvoiceDate > DATE '{C}' - INTERVAL 365 DAY) n_products_365,
            count(DISTINCT date_trunc('month', InvoiceDate)) FILTER (WHERE InvoiceDate > DATE '{C}' - INTERVAL 365 DAY) n_active_months_365,
            max(CASE WHEN Country = 'United Kingdom' THEN 1 ELSE 0 END) uk
          FROM p WHERE InvoiceDate < DATE '{C}' + INTERVAL 1 DAY GROUP BY cid
          HAVING max(InvoiceDate) > DATE '{C}' - INTERVAL 365 DAY) h
    LEFT JOIN (SELECT DISTINCT cid FROM p WHERE InvoiceDate >= DATE '{C}' + INTERVAL 1 DAY AND InvoiceDate < DATE '{C}' + INTERVAL 91 DAY) f USING (cid)
    LEFT JOIN (SELECT cid, count(DISTINCT Invoice) n_returns_365, -sum(amount) return_amount_365 FROM t
               WHERE is_return AND InvoiceDate < DATE '{C}' + INTERVAL 1 DAY AND InvoiceDate > DATE '{C}' - INTERVAL 365 DAY GROUP BY cid) r USING (cid)""")
con.execute(f"COPY ({' UNION ALL '.join(parts)} ORDER BY cutoff, cid) TO '{OUT}/retail_monthly.parquet' (FORMAT parquet)")
for r in con.execute(f"SELECT cutoff, count(*), round(avg(is_churn),3), round(median(recency_days)) FROM '{OUT}/retail_monthly.parquet' GROUP BY 1 ORDER BY 1").fetchall():
    print(*r)
