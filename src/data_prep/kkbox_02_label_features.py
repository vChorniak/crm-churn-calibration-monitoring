"""KKBox (WSDM Cup 2018): monthly churn labels and point-in-time features.

Labels follow the organisers' published labelling procedure (WSDMChurnLabeller.scala), applied to every
target month M instead of only the two months released for the competition:
  history     = transactions of month M-1 (as in the official labeller);
  candidates  = users whose last membership expiry (by the labeller's ordering of transactions) falls in M;
  future      = transactions after the cut-off C (last day of M-1) and no later than the end of M+1;
  churn       = no renewal, or first renewal >= 30 days after expiry.
Features use only transactions dated <= C.

Usage: python kkbox_02_label_features.py <work_dir from kkbox_01> <output_dir> <users_per_month, e.g. 100000>
Outputs: <output_dir>/kkbox_monthly.parquet, <output_dir>/kkbox_month_stats.csv and a validation line
comparing the February 2017 labels with the official train.csv.
"""
import duckdb, sys, os, datetime as dt
KK, OUT, N_PER_MONTH = sys.argv[1], sys.argv[2], int(sys.argv[3])
con = duckdb.connect()
con.execute(f"SET memory_limit='2GB'; SET temp_directory='{KK}/tmp'; SET threads=4;")
con.execute(f"CREATE VIEW tx AS SELECT *, CAST(plan_list_price AS VARCHAR)||CAST(payment_plan_days AS VARCHAR)||CAST(payment_method_id AS VARCHAR) AS sig FROM '{KK}/tx.parquet'")
con.execute(f"CREATE VIEW mem AS SELECT * FROM '{KK}/members.parquet'")

def ym_add(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1
def first(y, m): return y * 10000 + m * 100 + 1
def last(y, m):
    ny, nm = ym_add(y, m, 1)
    d = dt.date(ny, nm, 1) - dt.timedelta(days=1)
    return int(d.strftime('%Y%m%d'))
D = lambda col: f"strptime(CAST({col} AS VARCHAR), '%Y%m%d')::DATE"
ORD = "transaction_date {a}, sig {b}, is_cancel {a}, CASE WHEN is_cancel=1 THEN -membership_expire_date ELSE membership_expire_date END {a}"

months = [(2015, m) for m in range(2, 13)] + [(2016, m) for m in range(1, 13)] + [(2017, 1), (2017, 2)]
os.makedirs(f"{KK}/months", exist_ok=True)
for (y, m) in months:
    py, pm = ym_add(y, m, -1); fy, fm = ym_add(y, m, 1)
    hs, C, ms, me, fe = first(py, pm), last(py, pm), first(y, m), last(y, m), last(fy, fm)
    q = f"""
    WITH h AS (SELECT * FROM tx WHERE transaction_date BETWEEN {hs} AND {C}),
    lst AS (SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY msno ORDER BY {ORD.format(a='DESC', b='ASC')}) rn FROM h) WHERE rn = 1),
    cand AS (SELECT msno, membership_expire_date AS last_expire, payment_method_id, payment_plan_days,
                    plan_list_price, actual_amount_paid, is_auto_renew, is_cancel AS last_is_cancel
             FROM lst WHERE membership_expire_date BETWEEN {ms} AND {me}),
    f AS (SELECT t.msno, t.transaction_date, t.membership_expire_date, t.is_cancel,
                 row_number() OVER (PARTITION BY t.msno ORDER BY {ORD.format(a='ASC', b='DESC').replace('transaction_date','t.transaction_date').replace('is_cancel','t.is_cancel').replace('membership_expire_date','t.membership_expire_date').replace('sig','t.sig')}) rn
          FROM tx t SEMI JOIN cand c ON t.msno = c.msno
          WHERE t.transaction_date > {C} AND t.transaction_date <= {fe}),
    nc AS (SELECT msno, min(rn) rn_nc FROM f WHERE is_cancel = 0 GROUP BY msno),
    ncd AS (SELECT f.msno, f.transaction_date nc_date FROM f JOIN nc ON f.msno = nc.msno AND f.rn = nc.rn_nc),
    canc AS (SELECT f.msno, min(f.membership_expire_date) min_cexp FROM f LEFT JOIN nc USING (msno)
             WHERE f.is_cancel = 1 AND (nc.rn_nc IS NULL OR f.rn < nc.rn_nc) GROUP BY f.msno),
    lab AS (SELECT c.*, ncd.nc_date,
               CASE WHEN ncd.nc_date IS NULL THEN 1
                    WHEN date_diff('day', {D('least(c.last_expire, coalesce(canc.min_cexp, c.last_expire))')}, {D('ncd.nc_date')}) >= 30 THEN 1
                    ELSE 0 END AS is_churn
            FROM cand c LEFT JOIN ncd USING (msno) LEFT JOIN canc USING (msno)),
    hist AS (SELECT t.msno,
               date_diff('day', {D('max(t.transaction_date)')}, {D(C)}) AS recency_days,
               date_diff('day', {D('min(t.transaction_date)')}, {D(C)}) AS tx_tenure_days,
               count(*) AS n_tx_total,
               count(*) FILTER (WHERE t.transaction_date > {C} - 10000) AS n_tx_365,
               count(*) FILTER (WHERE t.transaction_date > {C} - 10000 AND t.is_cancel = 1) AS n_cancel_365,
               coalesce(sum(t.actual_amount_paid) FILTER (WHERE t.transaction_date > {C} - 10000), 0) AS amount_365,
               avg(t.is_auto_renew) FILTER (WHERE t.transaction_date > {C} - 10000) AS share_auto_365,
               avg(t.plan_list_price - t.actual_amount_paid) FILTER (WHERE t.transaction_date > {C} - 10000) AS mean_discount_365,
               count(DISTINCT t.payment_plan_days) AS n_distinct_plans
             FROM tx t SEMI JOIN cand c ON t.msno = c.msno
             WHERE t.transaction_date <= {C} GROUP BY t.msno)
    SELECT {y*100+m} AS month, l.msno, l.is_churn,
           l.payment_method_id, l.payment_plan_days, l.plan_list_price, l.actual_amount_paid,
           l.plan_list_price - l.actual_amount_paid AS last_discount, l.is_auto_renew, l.last_is_cancel,
           date_diff('day', {D(C)}, {D('l.last_expire')}) AS days_to_expire,
           hist.* EXCLUDE (msno),
           mem.city, CASE WHEN mem.bd BETWEEN 10 AND 80 THEN mem.bd END AS age, mem.gender, mem.registered_via,
           date_diff('day', try_strptime(CAST(mem.registration_init_time AS VARCHAR), '%Y%m%d')::DATE, {D(C)}) AS reg_tenure_days
    FROM lab l LEFT JOIN hist USING (msno) LEFT JOIN mem USING (msno)
    """
    outm = f"{KK}/months/{y*100+m}.parquet"
    if os.path.exists(outm):
        continue
    con.execute(f"CREATE OR REPLACE TEMP TABLE cur AS {q}")
    n, rate = con.execute("SELECT count(*), avg(is_churn) FROM cur").fetchone()
    print(y * 100 + m, n, round(rate, 4), flush=True)
    if (y, m) == (2017, 2):
        con.execute(f"COPY (SELECT msno, is_churn FROM cur) TO '{KK}/our_labels_201702.parquet' (FORMAT parquet)")
    con.execute(f"COPY (SELECT *, {n} AS n_candidates_month FROM cur ORDER BY hash(msno || '{y*100+m}') LIMIT {N_PER_MONTH}) TO '{outm}.tmp' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT {y*100+m} AS month, {n} AS n_candidates, {rate} AS churn_rate) TO '{KK}/months/{y*100+m}.stats.csv' (HEADER)")
    os.rename(f"{outm}.tmp", outm)

con.execute(f"COPY (SELECT * FROM '{KK}/months/*.parquet' ORDER BY month) TO '{OUT}/kkbox_monthly.parquet' (FORMAT parquet, COMPRESSION zstd)")
con.execute(f"COPY (SELECT * FROM read_csv_auto('{KK}/months/*.stats.csv') ORDER BY month) TO '{OUT}/kkbox_month_stats.csv' (HEADER)")
print("VALIDATION", con.execute(f"""
  SELECT count(*) AS n_official, count(o.msno) AS n_matched, avg(CASE WHEN o.is_churn = t.is_churn THEN 1.0 END) AS agreement,
         avg(t.is_churn) AS off_rate, avg(o.is_churn) AS our_rate
  FROM '{KK}/train_feb17.parquet' t LEFT JOIN '{KK}/our_labels_201702.parquet' o USING (msno)""").fetchall())
print("done")
