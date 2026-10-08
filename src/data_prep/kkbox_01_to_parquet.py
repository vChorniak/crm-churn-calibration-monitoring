"""KKBox (WSDM Cup 2018): convert the raw CSV files to Parquet for the labelling step.

Expected files in <raw_dir> (unpacked from the Kaggle competition "kkbox-churn-prediction-challenge"):
  transactions.csv, transactions_v2.csv, members_v3.csv, train.csv
(user_logs*.csv are not needed.)

Usage: python kkbox_01_to_parquet.py <raw_dir> <work_dir>
Outputs in <work_dir>: tx.parquet (transactions + transactions_v2, de-duplicated), members.parquet, train_feb17.parquet
"""
import os, sys
import duckdb

raw, work = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(work, "tmp"), exist_ok=True)
con = duckdb.connect()
con.execute(f"SET temp_directory='{work}/tmp'; SET preserve_insertion_order=false;")
cols = ("{'msno':'VARCHAR','payment_method_id':'SMALLINT','payment_plan_days':'SMALLINT','plan_list_price':'SMALLINT',"
        "'actual_amount_paid':'SMALLINT','is_auto_renew':'TINYINT','transaction_date':'INTEGER',"
        "'membership_expire_date':'INTEGER','is_cancel':'TINYINT'}")
con.execute(f"""COPY (SELECT * FROM read_csv('{raw}/transactions.csv', header=true, columns={cols})
                     UNION
                     SELECT * FROM read_csv('{raw}/transactions_v2.csv', header=true, columns={cols}))
               TO '{work}/tx.parquet' (FORMAT parquet, COMPRESSION zstd)""")
con.execute(f"COPY (SELECT * FROM read_csv_auto('{raw}/members_v3.csv', header=true)) TO '{work}/members.parquet' (FORMAT parquet, COMPRESSION zstd)")
con.execute(f"COPY (SELECT * FROM read_csv_auto('{raw}/train.csv', header=true)) TO '{work}/train_feb17.parquet' (FORMAT parquet)")
print(con.execute(f"SELECT count(*), count(DISTINCT msno), min(transaction_date), max(transaction_date) FROM '{work}/tx.parquet'").fetchall())
