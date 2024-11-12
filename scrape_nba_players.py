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

    # Loop through each page (1 to 11) and collect player profile links
    for page_num in range(11):
        try:
            # Wait until the page dropdown is present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select.DropDown_select__4pIg9"))
            )

            # Select the dropdown and choose the page number
            dropdown = driver.find_element(By.CSS_SELECTOR, "select.DropDown_select__4pIg9")
            dropdown.click()
            page_option = dropdown.find_element(By.XPATH, f"//option[@value='{page_num}']")
            page_option.click()
            print(f"Selected page {page_num + 1}")

            # Wait for the page content to load
            sleep(5)  # Adjust the sleep time if needed for slower loading pages

            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Extract player links on the current page
            players = soup.find_all("a", href=True, class_="Anchor_anchor__cSc3P")
            for player in players:
                if "/player/" in player["href"]:
                    full_link = "https://www.nba.com" + player["href"]
                    if full_link not in player_links:  # Avoid duplicates
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

    # Wait for the player's page to load
    try:
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '*[class*="PlayerSummary_mainInnerBio"]'))
        )
    except Exception as e:
        print(f"Error loading page for {player_url}: {e}")
        return {}

    if player_num == 0:
        try:
            sleep(2)  # Short delay to allow the cookies popup to load
            accept_button = driver.find_element(By.ID, "onetrust-accept-btn-handler")
            accept_button.click()
            print("Accepted cookies.")
        except Exception:
            print("Accept button not found.")

    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")
    player_data = {}

    # Extract team, jersey number, position, name, and surname
    try:
        main_info = soup.find("p", class_=re.compile(r"PlayerSummary_mainInnerInfo"))
        if main_info:
            # Split the text by '|' and take the first element, if it exists
            player_data["Team Info"] = main_info.text.strip().split('|')[0].strip()
            # Extract first name and last name
            first_name = soup.find_all("p", class_=re.compile(r"PlayerSummary_playerNameText"))[0].text.strip()
            last_name = soup.find_all("p", class_=re.compile(r"PlayerSummary_playerNameText"))[1].text.strip()
            player_data["Name"] = first_name + " " + last_name
    except AttributeError as e:
        print(f"Error extracting Name/Position for {player_url}: {e}")

    # Extract main stats (PPG, RPG, APG, PIE)
    stats_section = soup.find_all("div", class_=re.compile(r"PlayerSummary_playerStat"))
    for stat in stats_section:
        try:
            label = stat.find("p", class_=re.compile(r"PlayerSummary_playerStatLabel")).text.strip()
            value = stat.find("p", class_=re.compile(r"PlayerSummary_playerStatValue")).text.strip()
            player_data[label] = value
        except AttributeError:
            print(f"Stat label/value missing for {player_url}")

    # Extract additional player info like HEIGHT, WEIGHT, COUNTRY, etc.
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

    # Loop through each player's link to gather their data
    for player_num in range(len(player_links)):  # Limit to first 10 players for testing; remove for full scrape
        player_data = get_player_details(driver, player_links[player_num], player_num)
        players_data.append(player_data)
        print(f"Scraped data for: {player_data.get('Name', 'Unknown')}")

    driver.close()

    # Save to Excel
    df = pd.DataFrame(players_data)
    df.to_excel("nba_players_detailed.xlsx", index=False)
    print("Saved player data to nba_players_detailed.xlsx")


def merge_tables():


    # Load the existing files
    nba_stats = pd.read_excel("nba.xlsx")
    nba_players = pd.read_excel("nba_players_detailed.xlsx")

    # Ensure that the column names match for merging by player name
    # If necessary, rename columns to match exactly
    nba_stats.rename(columns={"Name": "Player Name"}, inplace=True)  # Change 'Player Name' if needed
    nba_players.rename(columns={"Name": "Player Name"}, inplace=True)  # Change 'Player Name' if needed

    # Merge the dataframes on the player name column
    merged_data = pd.merge(nba_stats, nba_players, on="Player Name", how="inner")

    # Save the merged dataframe to a new Excel file
    merged_data.to_excel("nba_merged.xlsx", index=False)
    print("Merged data saved to nba_merged.xlsx")


if __name__ == "__main__":
    scrape_nba_players()
    merge_tables()
