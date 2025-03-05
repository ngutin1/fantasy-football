import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime

# Define constants
POSITIONS = ['te']
YEARS = range(2020, 2025)  # 2020-2024
BASE_URL = "https://www.fantasypros.com/nfl/stats/{}.php"
OUTPUT_DIR = "fantasy_data"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0'
}

def scrape_data(position, year, data_range="full", week=None):
    """
    Scrapes fantasy football data from FantasyPros.
    
    Args:
        position: String - qb, rb, wr, or te
        year: Integer - year to scrape
        data_range: String - "full" for full season or "week" for weekly data
        week: Integer - week number (only used if data_range is "week")
        
    Returns:
        DataFrame with the scraped data
    """
    url = BASE_URL.format(position)
    params = {"year": year, "range": data_range}
    
    if data_range == "week" and week is not None:
        params["week"] = week
    
    print(f"Scraping {position.upper()} data for {year} ({data_range}{' week ' + str(week) if week else ''})")
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table
        table = soup.find('table', class_='table')
        if not table:
            print(f"No table found for {position} in {year} ({data_range}{' week ' + str(week) if week else ''})")
            return None
        
        # Parse the table headers
        table_headers = []
        for th in table.find('thead').find_all('th'):
            header_text = th.text.strip()
            table_headers.append(header_text if header_text else f"col_{len(table_headers)}")
        
        # Parse the table rows
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            row = []
            for td in tr.find_all('td'):
                # Extract player name and team if available
                if 'player-label' in td.get('class', []):
                    player_link = td.find('a', class_='player-name')
                    if player_link:
                        player_name = player_link.text.strip()
                        # Extract team (usually in parentheses after player name)
                        team_text = td.text.strip()
                        if '(' in team_text and ')' in team_text:
                            team = team_text[team_text.find('(')+1:team_text.find(')')]
                        else:
                            team = 'N/A'
                        row.append(player_name)
                        row.append(team)  # Add team as a separate column
                    else:
                        row.append(td.text.strip())
                        row.append('N/A')  # Add placeholder for team
                else:
                    row.append(td.text.strip())
            
            if row:  # Only add non-empty rows
                rows.append(row)
        
        # If we extracted team as a separate column, add it to headers
        if rows and len(rows[0]) > len(table_headers):
            table_headers.insert(1, 'Team')  # Add Team after Player name
        
        # Create a DataFrame
        df = pd.DataFrame(rows, columns=table_headers)
        
        # Add metadata columns
        df['Year'] = year
        df['Position'] = position.upper()
        if data_range == "week" and week is not None:
            df['Week'] = week
        else:
            df['Season'] = 'Full'
            
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {position} data for {year}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error scraping {position} data for {year}: {e}")
        return None

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrames to store all data
    full_season_data = []
    weekly_data = []
    
    # Scrape data for each position and year
    for position in POSITIONS:
        for year in YEARS:
            # Scrape full season data
            df_full = scrape_data(position, year)
            if df_full is not None:
                full_season_data.append(df_full)
                
                # Save individual position/year full season data
                output_file = os.path.join(OUTPUT_DIR, f"{position}_{year}_full_season.csv")
                df_full.to_csv(output_file, index=False)
                print(f"Saved {output_file}")
            
            # Scrape weekly data for each week (assuming 18 weeks max for newer seasons)
            max_weeks = 18 if year >= 2021 else 17  # 17 games before 2021, 18 after
            
            for week in range(1, max_weeks + 1):
                df_week = scrape_data(position, year, "week", week)
                if df_week is not None:
                    weekly_data.append(df_week)
                    
                    # Save individual position/year/week data
                    output_file = os.path.join(OUTPUT_DIR, f"{position}_{year}_week_{week}.csv")
                    df_week.to_csv(output_file, index=False)
                    print(f"Saved {output_file}")
                
                # Sleep to avoid overloading the server
                time.sleep(1)
            
            # Sleep between years
            time.sleep(2)
    
    # Combine and save all full season data
    if full_season_data:
        df_all_full = pd.concat(full_season_data, ignore_index=True)
        df_all_full.to_csv(os.path.join(OUTPUT_DIR, f"all_positions_full_season_{timestamp}.csv"), index=False)
        print(f"Saved combined full season data")
    
    # Combine and save all weekly data
    if weekly_data:
        df_all_weekly = pd.concat(weekly_data, ignore_index=True)
        df_all_weekly.to_csv(os.path.join(OUTPUT_DIR, f"all_positions_weekly_{timestamp}.csv"), index=False)
        print(f"Saved combined weekly data")
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"Script executed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()