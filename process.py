import pandas as pd

df = pd.read_excel("nba.xlsx")

# convert minute data to integer
df["Minute"] = df["Minute"].apply(lambda s: int(s.replace("'", "")))

# obtain field goals attempted and made from fractional column field goals
df["Field Goals Attempted"] = df["Field Goals"].apply(lambda s: int(s.split("/")[1]))
df["Field Goals Made"] = df["Field Goals"].apply(lambda s: int(s.split("/")[0]))
df = df.drop(columns=["Field Goals"])
df["Fields Goal Percentage"] = df["Fields Goal Percentage"].apply(lambda s: int(s.replace("%", "")))
df = df.rename(columns={"Fields Goal Percentage": "Field Goals Percentage"})  # just fixing a typo

# obtain free throws attempted and made from fractional column free throws
df["Free Throws Attempted"] = df["Free Throws"].apply(lambda s: int(s.split("/")[1]))
df["Free Throws Made"] = df["Free Throws"].apply(lambda s: int(s.split("/")[0]))
df = df.drop(columns=["Free Throws"])
df["Free Throw Percentage"] = df["Free Throw Percentage"].apply(lambda s: int(s.replace("%", "")))
df = df.rename(columns={"Free Throw Percentage": "Free Throws Percentage"})  # just fixing a typo

# obtain 3 pointers attempted and made from fractional column 3 pointers
df["3 Pointers Attempted"] = df["3 Pointers"].apply(lambda s: int(s.split("/")[1]))
df["3 Pointers Made"] = df["3 Pointers"].apply(lambda s: int(s.split("/")[0]))
df = df.drop(columns=["3 Pointers"])
df["3 Pointers Percentage"] = df["3 Pointers Percentage"].apply(lambda s: int(s.replace("%", "")))

# normalize some data to per match, assuming a player plays 20mins on average in a single match
average_minute = 20.0
for c in ["Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers", "Offensive Rebounds", "Defensive Rebounds", "Field Goals Attempted", "Field Goals Made", "Free Throws Attempted", "Free Throws Made", "3 Pointers Attempted", "3 Pointers Made"]:
    c2 = c + " Per Match"
    df[c2] = df[c] / (df["Minute"] / average_minute)
    # df = df.drop(columns=[c])

# add team statistics
for c in ["Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers", "Offensive Rebounds", "Defensive Rebounds", "Field Goals Attempted", "Field Goals Made", "Free Throws Attempted", "Free Throws Made", "3 Pointers Attempted", "3 Pointers Made"]:
    c2 = c + " Team"
    c3 = c + " Contribution"
    df[c2] = df.groupby(["Match ID", "Team"])[c].transform("sum")
    df[c3] = df[c] / df[c2]
df = df.drop(columns=["Match ID", "Team"])

# remove players with less than 10 minutes played
df = df.drop(df[df["Minute"] < 5].index)

# TODO: get player-specific attributes and delete player name column

# delete unnecessary columns
df = df.drop(columns=["Unnamed: 0", "Minute"])
# remove fouls since mostly they will be 5-6 and probably will have no effect (?)
df = df.drop(columns=["Fouls"])
# remove performace rating
df = df.drop(columns=["Performance Rating"])

print(df.columns)

# df = df[
#     [
#         "Name", 
#         "Position", 
#         "Points", "Points Per Match", "Points Team", 
#         "Rebounds", "Rebounds Per Match", "Rebounds Team", "Offensive Rebounds", "Offensive Rebounds Per Match", "Offensive Rebounds Team", "Defensive Rebounds", "Defensive Rebounds Per Match", "Defensive Rebounds Team",
#         "Assists", "Assists Per Match", "Assists Team",
#         "Steals", "Steals Per Match", "Steals Team",
#         "Blocks", "Blocks Per Match", "Blocks Team",
#         "Turnovers", "Turnovers Per Match", "Turnovers Team",
#         "Field Goals Attempted", "Field Goals Attempted Per Match", "Field Goals Attempted Team", "Field Goals Made", "Field Goals Made Per Match", "Field Goals Made Team", "Field Goals Percentage",
#         "Free Throws Attempted", "Free Throws Attempted Per Match", "Free Throws Attempted Team", "Free Throws Made", "Free Throws Made Per Match", "Free Throws Made Team", "Free Throws Percentage",
#         "3 Pointers Attempted", "3 Pointers Attempted Per Match", "3 Pointers Attempted Team", "3 Pointers Made", "3 Pointers Made Per Match", "3 Pointers Made Team", "3 Pointers Percentage",
#     ]
# ]

df = df[
    [
        "Name", 
        "Position", 
        "Points Per Match", "Points Contribution", 
        "Rebounds Per Match", "Rebounds Contribution", "Offensive Rebounds Per Match", "Offensive Rebounds Contribution", "Defensive Rebounds Per Match", "Defensive Rebounds Contribution",
        "Assists Per Match", "Assists Contribution",
        "Steals Per Match", "Steals Contribution",
        "Blocks Per Match", "Blocks Contribution",
        "Turnovers Per Match", "Turnovers Contribution",
        "Field Goals Attempted Per Match", "Field Goals Attempted Contribution", "Field Goals Made Per Match", "Field Goals Made Contribution", "Field Goals Percentage",
        "Free Throws Attempted Per Match", "Free Throws Attempted Contribution", "Free Throws Made Per Match", "Free Throws Made Contribution", "Free Throws Percentage",
        "3 Pointers Attempted Per Match", "3 Pointers Attempted Contribution", "3 Pointers Made Per Match", "3 Pointers Made Contribution", "3 Pointers Percentage",
    ]
]

df.to_excel("nba.xlsx", index=False)

