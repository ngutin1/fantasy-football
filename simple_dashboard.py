import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Simple Fantasy Football Volatility Dashboard",
    page_icon="🏈",
    layout="wide"
)

st.title("Fantasy Football Volatility Dashboard")
st.write("Analyze player volatility for your SuperFlex, 3WR, PPR, TE Premium league")

# Function to calculate volatility score
def calculate_volatility_score(game_scores, average_ppg):
    """
    Calculate volatility score based on game-by-game performance vs PPG
    Positive score = player tends to exceed their average (good volatility)
    Negative score = player tends to underperform their average (bad volatility)
    """
    if not isinstance(game_scores, (list, np.ndarray)) or len(game_scores) == 0:
        return 0
    
    volatility_score = 0
    total_games = len(game_scores)
    
    # Calculate positive and negative deviations
    for score in game_scores:
        deviation = score - average_ppg
        
        # Reward exceeding average, penalize falling below
        if deviation > 0:
            # Positive deviation (exceeded average)
            volatility_score += (deviation / average_ppg) * 0.5  # 50% weight for exceeding
        else:
            # Negative deviation (below average)
            volatility_score += (deviation / average_ppg) * 1.0  # 100% weight for underperforming
    
    # Normalize by games played for fair comparison
    return volatility_score / total_games

# Scan the fantasy_data directory to identify available data
def scan_fantasy_data():
    years = []
    positions = {}
    
    if not os.path.exists("fantasy_data"):
        st.error("fantasy_data folder not found. Please create it and add your CSV files.")
        return years, positions
    
    files = glob.glob("fantasy_data/*.csv")
    
    for file in files:
        filename = os.path.basename(file)
        parts = filename.replace('.csv', '').split('_')
        
        if len(parts) >= 2:
            position = parts[0].upper()  # Convert to uppercase for consistency
            year = parts[1]
            
            if year not in years:
                years.append(year)
            
            if year not in positions:
                positions[year] = []
            
            if position not in positions[year]:
                positions[year].append(position)
    
    return sorted(years), positions

# Get player data for a specific year and position
def get_player_data(year, position):
    # Get all files for this year and position
    pattern = f"fantasy_data/{position.lower()}_{year}_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        st.error(f"No data files found for {position} in {year}")
        return None, None
    
    # Identify weekly files and season files
    weekly_files = [f for f in files if 'season' not in f.lower()]
    
    # Load all weekly files and get player data
    player_data = {}
    
    for file in weekly_files:
        try:
            df = pd.read_csv(file)
            
            # Determine if we need to swap Player and Team columns
            # Check 2-3 rows to see if Player column contains likely team names (2-3 letters)
            if 'Player' in df.columns and 'Team' in df.columns:
                sample_players = df['Player'].head(3).tolist()
                sample_teams = df['Team'].head(3).tolist()
                
                # If Player column has mostly short strings (likely team abbreviations)
                # and Team column has longer strings (likely player names)
                if all(len(str(p)) <= 3 for p in sample_players if pd.notna(p)) and \
                   any(len(str(t)) > 3 for t in sample_teams if pd.notna(t)):
                    # Swap columns
                    df = df.rename(columns={'Team': 'PlayerName', 'Player': 'TeamName'})
                    df = df.rename(columns={'PlayerName': 'Player', 'TeamName': 'Team'})
            
            # Find the fantasy points column
            fpts_col = None
            for col in df.columns:
                if 'PTS' in col.upper() or 'FP' in col.upper():
                    fpts_col = col
                    break
            
            if not fpts_col:
                st.warning(f"No fantasy points column found in {os.path.basename(file)}")
                continue
            
            # Extract week number from filename
            week = os.path.basename(file).split('_')[-1].replace('.csv', '')
            try:
                week = int(week)
            except:
                week = os.path.basename(file).split('_')[-2] + '_' + week
            
            # Add player scores to data dictionary
            for _, row in df.iterrows():
                player_name = row['Player']
                if pd.notna(player_name) and player_name != '':
                    if player_name not in player_data:
                        player_data[player_name] = {'weekly_scores': {}}
                    
                    # Add weekly score if it's a valid number
                    try:
                        score = float(row[fpts_col])
                        player_data[player_name]['weekly_scores'][week] = score
                    except:
                        pass
        
        except Exception as e:
            st.warning(f"Error processing file {os.path.basename(file)}: {str(e)}")
    
    # Calculate averages and volatility
    for player, data in player_data.items():
        if data['weekly_scores']:
            scores = list(data['weekly_scores'].values())
            data['avg_ppg'] = np.mean(scores)
            data['volatility'] = calculate_volatility_score(scores, data['avg_ppg'])
            data['games_played'] = len(scores)
    
    # Convert to DataFrame for display
    if player_data:
        player_df = pd.DataFrame([
            {
                'Player': player,
                'Avg PPG': data['avg_ppg'],
                'Volatility': data['volatility'],
                'Games Played': data['games_played']
            }
            for player, data in player_data.items()
            if 'avg_ppg' in data
        ])
        
        # Sort by Avg PPG
        player_df = player_df.sort_values('Avg PPG', ascending=False)
        
        return player_df, player_data
    
    return None, None

# Plot player volatility
def plot_player_volatility(player_name, player_data):
    if player_name not in player_data:
        st.error(f"No data found for {player_name}")
        return None
    
    data = player_data[player_name]
    if not data['weekly_scores']:
        st.error(f"No weekly scores found for {player_name}")
        return None
    
    # Sort weeks numerically if possible
    try:
        weeks = sorted(data['weekly_scores'].keys(), key=int)
    except:
        weeks = sorted(data['weekly_scores'].keys())
    
    scores = [data['weekly_scores'][week] for week in weeks]
    avg_ppg = data['avg_ppg']
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(weeks, scores, color='lightblue', alpha=0.7)
    
    # Add average line
    ax.axhline(y=avg_ppg, color='red', linestyle='--', label=f'Average: {avg_ppg:.1f}')
    
    # Color bars based on performance vs average
    for i, bar in enumerate(bars):
        if scores[i] > avg_ppg:
            bar.set_color('green')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)
    
    # Add labels and title
    ax.set_xlabel('Week')
    ax.set_ylabel('Fantasy Points')
    ax.set_title(f"{player_name} Weekly Performance (Volatility: {data['volatility']:.3f})")
    
    # Add legend
    ax.legend()
    
    # Add data labels on bars
    for i, score in enumerate(scores):
        ax.text(i, score + 0.5, f'{score:.1f}', ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Main dashboard layout
def main():
    # Scan available data
    years, positions = scan_fantasy_data()
    
    if not years:
        st.error("No data files found in fantasy_data folder. Please add your CSV files.")
        
        # Show expected file format
        st.write("Expected file naming format:")
        st.code("position_year_week_number.csv  (e.g., qb_2023_week_1.csv)")
        st.code("position_year_full_season.csv  (e.g., qb_2023_full_season.csv)")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    selected_year = st.sidebar.selectbox("Select Year", years)
    
    available_positions = positions.get(selected_year, [])
    if not available_positions:
        st.error(f"No position data found for {selected_year}")
        return
    
    selected_position = st.sidebar.selectbox("Select Position", sorted(available_positions))
    
    # Get player data
    player_df, player_data = get_player_data(selected_year, selected_position)
    
    if player_df is None or player_df.empty:
        st.error(f"No player data found for {selected_position} in {selected_year}")
        return
    
    # Show player data table
    st.header(f"{selected_year} {selected_position} - Player Volatility")
    
    # Format the dataframe
    display_df = player_df.copy()
    display_df['Avg PPG'] = display_df['Avg PPG'].map('{:.1f}'.format)
    display_df['Volatility'] = display_df['Volatility'].map('{:.3f}'.format)
    
    # Min games filter
    min_games = st.slider("Minimum Games Played", 1, 17, 4)
    filtered_df = display_df[display_df['Games Played'] >= min_games]
    
    if filtered_df.empty:
        st.warning(f"No players with at least {min_games} games")
        return
    
    # Show the data table
    st.dataframe(filtered_df, use_container_width=True)
    
    # Select player for detailed view
    st.header("Player Volatility Breakdown")
    selected_player = st.selectbox("Select Player", filtered_df['Player'].tolist())
    
    # Plot player volatility
    if selected_player:
        fig = plot_player_volatility(selected_player, player_data)
        if fig:
            st.pyplot(fig)
        
        # Show raw weekly data
        if selected_player in player_data:
            data = player_data[selected_player]
            
            if data['weekly_scores']:
                st.subheader("Weekly Scores")
                
                # Sort weeks
                try:
                    weeks = sorted(data['weekly_scores'].keys(), key=int)
                except:
                    weeks = sorted(data['weekly_scores'].keys())
                
                scores = [data['weekly_scores'][week] for week in weeks]
                
                # Create weekly data table
                weekly_df = pd.DataFrame({
                    'Week': weeks,
                    'FPTS': scores,
                    'Average': [data['avg_ppg']] * len(weeks),
                    'Deviation': [score - data['avg_ppg'] for score in scores]
                })
                
                # Calculate weekly volatility
                weekly_df['Volatility Contribution'] = weekly_df.apply(
                    lambda row: (row['Deviation'] / row['Average']) * 0.5 if row['Deviation'] > 0 
                    else (row['Deviation'] / row['Average']) * 1.0,
                    axis=1
                )
                
                # Format for display
                display_weekly = weekly_df.copy()
                display_weekly['FPTS'] = display_weekly['FPTS'].map('{:.1f}'.format)
                display_weekly['Average'] = display_weekly['Average'].map('{:.1f}'.format)
                display_weekly['Deviation'] = display_weekly['Deviation'].map('{:.1f}'.format)
                display_weekly['Volatility Contribution'] = display_weekly['Volatility Contribution'].map('{:.3f}'.format)
                
                st.dataframe(display_weekly, use_container_width=True)

if __name__ == "__main__":
    main()