import pandas as pd

#create long short returns with ff3 data 
#this is the script that creates long_short_returns_ff3

#load ff3 data and long short returns data
ls_df = pd.read_csv("Portfolio/portfolio_results.csv", parse_dates=["date"])
ff3 = pd.read_csv("Portfolio/F-F_Research_Data_Factors_daily.CSV", skiprows=4)
ff3 = ff3[pd.to_numeric(ff3.iloc[:, 0], errors="coerce").notna()]


#Parse date from YYYYMMDD integer format
ff3["date"] = pd.to_datetime(ff3.iloc[:, 0].astype(str), format="%Y%m%d")
ff3 = ff3.drop(columns=ff3.columns[0])

#Clean FF3 column names
ff3.columns = ff3.columns.str.strip().str.lower().str.replace("-", "_")

#convert to decimals to match return series
ff3[["mkt_rf", "smb", "hml", "rf"]] = ff3[["mkt_rf", "smb", "hml", "rf"]] / 100

#Drop the placeholder NaN columns from long_short_returns
ls_df = ls_df.drop(columns=["mkt_rf", "smb", "hml", "rf"], errors="ignore")

#Merge
merged = ls_df.merge(ff3[["date", "mkt_rf", "smb", "hml", "rf"]], on="date", how="left")

#Save
merged.to_csv("Portfolio/long_short_returns_ff3.csv", index=False)

print(f"Rows: {len(merged)}")
print(f"FF3 matched: {merged['mkt_rf'].notna().sum()} / {len(merged)}")
print(merged.head())