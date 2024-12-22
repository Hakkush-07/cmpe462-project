from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import re

from selenium.webdriver.common.action_chains import ActionChains

def get_chrome_options():
    chrome_options = Options()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    return chrome_options

def get_player_links():
    chrome_options = get_chrome_options()
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.nba.com/players")
    driver.implicitly_wait(20)

    player_links = []

    for page_num in range(11):
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select.DropDown_select__4pIg9"))
            )

            dropdown = driver.find_element(By.CSS_SELECTOR, "select.DropDown_select__4pIg9")
            dropdown.click()
            page_option = dropdown.find_element(By.XPATH, f"//option[@value='{page_num}']")
            page_option.click()
            print(f"Selected page {page_num + 1}")
            sleep(5)

            soup = BeautifulSoup(driver.page_source, "html.parser")

            players = soup.find_all("a", href=True, class_="Anchor_anchor__cSc3P")
            for player in players:
                if "/player/" in player["href"]:
                    full_link = "https://www.nba.com" + player["href"]
                    if full_link not in player_links:
                        player_links.append(full_link)

        except Exception as e:
            print(f"Error loading page {page_num + 1}: {e}")
            break

    driver.close()
    print(f"There are {len(player_links)} players.")
    return player_links


from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re


def get_player_details(driver, player_url, player_num):
    driver.get(player_url)
    try:
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '*[class*="PlayerSummary_mainInnerBio"]'))
        )
    except Exception as e:
        print(f"Error loading page for {player_url}: {e}")
        return {}

    if player_num == 0:
        try:
            sleep(2)
            accept_button = driver.find_element(By.ID, "onetrust-accept-btn-handler")
            accept_button.click()
            print("Accepted cookies.")
        except Exception:
            print("Accept button not found.")

    soup = BeautifulSoup(driver.page_source, "html.parser")
    player_data = {}

    try:
        main_info = soup.find("p", class_=re.compile(r"PlayerSummary_mainInnerInfo"))
        if main_info:
            player_data["Team Info"] = main_info.text.strip().split('|')[0].strip()
            first_name = soup.find_all("p", class_=re.compile(r"PlayerSummary_playerNameText"))[0].text.strip()
            last_name = soup.find_all("p", class_=re.compile(r"PlayerSummary_playerNameText"))[1].text.strip()
            player_data["Name"] = first_name + " " + last_name
    except AttributeError as e:
        print(f"Error extracting Name/Position for {player_url}: {e}")

    stats_section = soup.find_all("div", class_=re.compile(r"PlayerSummary_playerStat"))
    for stat in stats_section:
        try:
            label = stat.find("p", class_=re.compile(r"PlayerSummary_playerStatLabel")).text.strip()
            value = stat.find("p", class_=re.compile(r"PlayerSummary_playerStatValue")).text.strip()
            player_data[label] = value
        except AttributeError:
            print(f"Stat label/value missing for {player_url}")


    info_section = soup.find_all("div", class_=re.compile(r"PlayerSummary_playerInfo"))
    for info in info_section:
        try:
            label = info.find("p", class_=re.compile(r"PlayerSummary_playerInfoLabel")).text.strip()
            value = info.find("p", class_=re.compile(r"PlayerSummary_playerInfoValue")).text.strip()
            player_data[label] = value
        except AttributeError:
            print(f"Info label/value missing for {player_url}")

    return player_data


def scrape_nba_players():
    player_links = get_player_links()
    chrome_options =  get_chrome_options()

    driver = webdriver.Chrome(options=chrome_options)
    players_data = []

    for player_num in range(len(player_links)):
        player_data = get_player_details(driver, player_links[player_num], player_num)
        players_data.append(player_data)
        print(f"Scraped data for: {player_data.get('Name', 'Unknown')}")

    driver.close()

    df = pd.DataFrame(players_data)
    df.to_excel("nba_players_detailed.xlsx", index=False)
    print("Saved player data to nba_players_detailed.xlsx")


def merge_tables():


    nba_stats = pd.read_excel("nba.xlsx")
    nba_players = pd.read_excel("nba_players_detailed.xlsx")
    nba_stats.rename(columns={"Name": "Player Name"}, inplace=True)
    nba_players.rename(columns={"Name": "Player Name"}, inplace=True)

    merged_data = pd.merge(nba_stats, nba_players, on="Player Name", how="inner")

    merged_data.to_excel("nba_merged.xlsx", index=False)
    print("Merged data saved to nba_merged.xlsx")


if __name__ == "__main__":
    scrape_nba_players()
    merge_tables()
