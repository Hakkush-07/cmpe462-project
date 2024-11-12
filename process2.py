import pandas as pd

df = pd.read_excel("nba_merged.xlsx")

df = df.drop(columns=["Team Info", "PPG", "RPG", "APG", "PIE", "COUNTRY", "LAST ATTENDED", "BIRTHDATE", "DRAFT", "EXPERIENCE"])

df["Age"] = df["AGE"].apply(lambda s: int(s[:2]))
df["Height"] = df["HEIGHT"].apply(lambda s: int(100 * float(s.split("(")[1].split("m)")[0])))
df["Weight"] = df["WEIGHT"].apply(lambda s: int(s.split("(")[1].split("kg)")[0]) if len(s) > 2 else 93)

df = df.drop(columns=["HEIGHT", "WEIGHT", "AGE"])

df.to_excel("nba.xlsx", index=False)



