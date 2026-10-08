"""Online Retail II (UCI, DOI 10.24432/C5CG6D): convert both worksheets of online_retail_II.xlsx into one
de-duplicated Parquet file (the two sheets overlap in December 2010).

Usage: python retail_01_xlsx_to_parquet.py <online_retail_II.xlsx> <work_dir>
Output: <work_dir>/retail.parquet
"""
import csv, os, sys
import duckdb, openpyxl

src, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
wb = openpyxl.load_workbook(src, read_only=True)
csvs = []
for i, ws in enumerate(wb.worksheets):
    out = os.path.join(work, f"retail_sheet{i}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        for row in ws.iter_rows(values_only=True):
            w.writerow(row)
    csvs.append(out)
    print("sheet", ws.title, "->", out)
cols = ("{'Invoice':'VARCHAR','StockCode':'VARCHAR','Description':'VARCHAR','Quantity':'INTEGER','InvoiceDate':'TIMESTAMP',"
        "'Price':'DOUBLE','Customer ID':'DOUBLE','Country':'VARCHAR'}")
con = duckdb.connect()
con.execute(f"COPY (SELECT DISTINCT * FROM read_csv({csvs}, header=true, columns={cols})) "
            f"TO '{os.path.join(work, 'retail.parquet')}' (FORMAT parquet)")
print(con.execute(f"SELECT count(*), min(InvoiceDate), max(InvoiceDate) FROM '{os.path.join(work, 'retail.parquet')}'").fetchall())
