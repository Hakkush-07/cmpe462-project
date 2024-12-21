import pandas as pd
def read_and_process():
    df = pd.read_excel("nba.xlsx")
    del df["Player Name"]
    df = df[[c for c in df.columns if c != "Position"] + ["Position"]]
    df["Position"] = df["Position"].apply(lambda x: "Center" if x.startswith("Center") else "Guard" if x.startswith("Guard") else "Forward")
    df.fillna(0,inplace=True)
    df["Position"] = df["Position"].apply(lambda x: ["Guard", "Forward", "Center"].index(x))
    return df

