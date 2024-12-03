from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd
from os import listdir

def test():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    URL = "https://www.sofavpn.com/tournament/basketball/usa/nba/132#id:65360,tab:matches"
    driver.get(URL)
    driver.implicitly_wait(20)
    sleep(1)
    html = driver.page_source
    
    with open("test.html", "w+") as file:
        file.write(html)
    print(html)
    driver.close()

# create matches folder before running this, and after that combine all results
def get_matches():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.sofavpn.com/tournament/basketball/usa/nba/132#id:65360,tab:matches")
    driver.implicitly_wait(20)
    sleep(1)
    html = driver.page_source

    soup = BeautifulSoup(html, "lxml")
    matches = []
    for match in soup.find_all("a", {"data-testid": "event_cell"}):
        matches.append((match["data-id"], match["href"]))
    
    with open("matches/matches-0.txt", "w+") as file:
        for j, link in matches:
            file.write(f"{j} {link}\n")
    
    i = 1
    while True:
        # click previous button
        driver.find_elements(By.CSS_SELECTOR, "button[class='Button iCnTrv']")[0].click()

        driver.implicitly_wait(20)
        sleep(1)

        html = driver.page_source

        soup = BeautifulSoup(html, "lxml")
        matches = []
        for match in soup.find_all("a", {"data-testid": "event_cell"}):
            matches.append((match["data-id"], match["href"]))
        
        with open(f"matches/matches-{i}.txt", "w+") as file:
            for j, link in matches:
                file.write(f"{j} {link}\n")
        i += 1
        if i > 33:
            break
    
    driver.close()

def combine_match_txt():
    matches = []
    for filename in listdir("matches"):
        with open(f"matches/{filename}", "r+") as file:
            for m in file.read().splitlines():
                matches.append(m)
    with open("nba.txt", "w+") as file:
        file.write("\n".join(matches) + "\n")

def stat_parser(html_team):
    soup = BeautifulSoup(html_team, "lxml")
    players = []
    stats = []
    teams = []
    # team_name = soup.find("div", {"class": "Box Flex gUNQxL izMxjT"}).find("img")["alt"]
    team1 = soup.find_all("img", {"class": "Img jbaYme"})[0]
    team2 = soup.find_all("img", {"class": "Img jbaYme"})[1]
    team_name1 = team1["alt"]
    team_name2 = team2["alt"]
    team_id1 = team1["src"]
    team_id2 = team2["src"]
    
    for player in soup.find("div", {"class": "Box Flex eYBhhw iWGVcA"}).find_all("div", {"class": "Box Flex ggRYVx cQgcrM"}):
        players.append((player.find("img")["alt"], player.find("span", {"class": "Text cMicsT"}).text))
    for a in soup.find("div", {"class": "Box Flex iLOrTM fFmyKd"}).find_all("div", {"class": "Box Flex ggRYVx cQgcrM"}):
        stats.append([div.find("div").text for div in a.find_all("div", {"display": "flex"})])
    for t in soup.find("div", {"class": "Box Flex eYBhhw iWGVcA"}).find_all("div", {"class": "Box Flex eJCdjm fRroAj"}):
        teams.append(team_name1 if t.find("img")["src"] == team_id1 else team_name2)
    dct = {}
    for p, n, s in zip(players, teams, stats):
        dct[p] = [n] + s
    return dct
    
def get_match_info(link):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(990, 1500)
    driver.get("https://www.sofavpn.com" + link + ",tab:box_score")
    driver.implicitly_wait(20)
    sleep(1)
    driver.save_screenshot("test.png")

    # click detailed stats button
    driver.find_elements(By.CSS_SELECTOR, "span[class=slider]")[0].click()
    sleep(0.1)
    # driver.save_screenshot("test1.png")

    # get both team stats
    html = driver.page_source
    driver.close()

    return stat_parser(html)

# create nba folder before running, and after combine all excels
def f():
    with open("nba.txt", "r+") as file:
        for match_id, match_link in map(lambda x: x.split(), file.read().splitlines()):
            data = []
            for k, v in get_match_info(match_link).items():
                player_name, player_position = k
                team_name = v[0]
                rest = v[1:]
                data.append([match_id, team_name, player_name, player_position] + rest)
            df = pd.DataFrame(data)
            df.columns = ["Match ID", "Team", "Name", "Position", "Minute", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Fouls", "Turnovers", "Offensive Rebounds", "Defensive Rebounds", "Field Goals", "Fields Goal Percentage", "Free Throws", "Free Throw Percentage", "3 Pointers", "3 Pointers Percentage", "Performance Rating"]
            df.to_excel(f"nba/m-{match_id}.xlsx")

def combine_excels():
    dfs = []
    for file in listdir("nba"):
        df = pd.read_excel(f"nba/{file}")
        dfs.append(df)
    dfx = pd.concat(dfs)
    dfx.to_excel("nba.xlsx", index=False)

if __name__ == "__main__":
    # get_matches()
    # combine_match_txt()
    # get_match_info("/basketball/match/brooklyn-nets-sacramento-kings/ntbsLtb#id:12696573")
    # f()
    combine_excels()






