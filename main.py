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

        html = driver.page_source

        soup = BeautifulSoup(html, "lxml")
        matches = []
        for match in soup.find_all("a", {"data-testid": "event_cell"}):
            matches.append((match["data-id"], match["href"]))
        
        with open(f"matches/matches-{i}.txt", "w+") as file:
            for j, link in matches:
                file.write(f"{j} {link}\n")
        i += 1
        if i > 20:
            break
    
    driver.close()

def stat_parser(html_team):
    soup = BeautifulSoup(html_team, "lxml")
    players = []
    stats = []
    team_name = soup.find("div", {"class": "Box Flex gUNQxL izMxjT"}).find("img")["alt"]
    for player in soup.find("div", {"class": "Box Flex eYBhhw iWGVcA"}).find_all("div", {"class": "Box Flex ggRYVx cQgcrM"}):
        players.append((player.find("img")["alt"], player.find("span", {"class": "Text cMicsT"}).text))
    for a in soup.find("div", {"class": "Box Flex aWUUI eAsfYD"}).find_all("div", {"class": "Box Flex ggRYVx cQgcrM"}):
        stats.append([div.find("div").text for div in a.find_all("div", {"display": "flex"})])
    dct = {}
    for p, s in zip(players, stats):
        dct[p] = s
    return dct, team_name
    
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
    # driver.save_screenshot("test1.png")

    # get the first team stats
    html_a = driver.page_source

    # click on the second team
    driver.find_elements(By.CSS_SELECTOR, "div[data-testid=right]")[0].click()
    # driver.save_screenshot("test2.png")

    # get the second team stats
    html_b = driver.page_source
    driver.close()

    a, team_a = stat_parser(html_a)
    b, team_b = stat_parser(html_b)
    return a, team_a, b, team_b

# create nba folder before running, and after combine all excels
def f():
    with open("nba.txt", "r+") as file:
        for match_id, match_link in map(lambda x: x.split(), file.read().splitlines()):
            data = []
            a, team_a, b, team_b = get_match_info(match_link)
            for k, v in a.items():
                player_name, player_position = k
                data.append([match_id, team_a, player_name, player_position] + v)
            for k, v in b.items():
                player_name, player_position = k
                data.append([match_id, team_b, player_name, player_position] + v)
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
    # a, team_a, b, team_b = get_match_info("/basketball/match/alba-berlin-panathinaikos-bc/ivbsqvb#id:12544784")
    # print(a)
    # print(b)
    # print(team_a)
    # print(team_b)

    # f()

    combine_excels()





