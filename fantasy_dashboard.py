import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(
    page_title="Fantasy Football Volatility Dashboard",
    page_icon="🏈",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f3f4f6;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        color: #4b5563;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .highlight-positive {
        color: #047857;
    }
    .highlight-negative {
        color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">Fantasy Football Volatility Dashboard</div>', unsafe_allow_html=True)
st.markdown('Analyze player performance with advanced volatility metrics for your SuperFlex, 3WR, PPR, TE Premium league')

# Define position multipliers for league settings
POSITION_MULTIPLIERS = {
    'QB': 1.4,  # Higher due to SuperFlex
    'RB': 1.0,  # Baseline
    'WR': 1.15, # Boosted for 3WR requirement
    'TE': 1.25  # Premium scoring
}

# Define position scarcity factors
POSITION_SCARCITY = {
    'QB': 1.3,
    'RB': 1.15,
    'WR': 1.2,
    'TE': 1.25
}

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

# Function to calculate consistency score
def calculate_consistency_score(game_scores, average_ppg):
    """
    Calculate consistency score - a measure of how often a player
    performs close to their average (standard deviation based)
    """
    if not isinstance(game_scores, (list, np.ndarray)) or len(game_scores) < 2:
        return 0
    
    # Calculate standard deviation
    std_dev = np.std(game_scores)
    
    # Calculate coefficient of variation (CV) - lower is better
    cv = std_dev / average_ppg if average_ppg > 0 else 0
    
    # Transform to 0-1 scale where 1 is perfectly consistent
    # Most fantasy players have CV between 0.3 and 0.9
    consistency_score = max(0, min(1, 1 - (cv / 0.9)))
    
    return consistency_score

# Function to calculate player value
def calculate_player_value(player_data):
    """
    Calculate player fantasy value based on metrics and league settings
    """
    position = player_data['position']
    ppg = player_data['ppg']
    game_scores = player_data.get('game_scores', [])
    games_played = player_data.get('games_played', len(game_scores) if game_scores else 0)
    reliability = player_data.get('reliability', 1.0)
    
    # Calculate volatility metrics
    volatility_score = calculate_volatility_score(game_scores, ppg)
    consistency_score = calculate_consistency_score(game_scores, ppg)
    
    # Base value using position multiplier
    value = ppg * POSITION_MULTIPLIERS.get(position, 1.0)
    
    # Apply volatility adjustment
    volatility_adjustment = (volatility_score * 0.4) + (consistency_score * 0.6)
    value *= (1 + volatility_adjustment)
    
    # Apply reliability adjustment
    reliability_score = (games_played / 17) * reliability
    value *= (0.7 + (0.3 * reliability_score))
    
    # Apply position scarcity
    value *= POSITION_SCARCITY.get(position, 1.0)
    
    return value

# Function to assign tier based on value
def assign_value_tier(value):
    if value >= 25:
        return "Elite"
    elif value >= 20:
        return "High-End Starter"
    elif value >= 15:
        return "Solid Starter"
    elif value >= 10:
        return "Flex Worthy"
    else:
        return "Bench/Depth"

# Function to load and process fantasy data
@st.cache_data
def load_fantasy_data(data_folder="fantasy_data"):
    """
    Load fantasy football data from the specified folder
    Expected format: {position}_{year}_{week/full season}.csv
    """
    all_data = {}
    weekly_data = {}
    seasonal_data = {}
    
    # Get list of all CSV files
    try:
        file_pattern = os.path.join(data_folder, "*.csv")
        csv_files = glob.glob(file_pattern)
        
        if not csv_files:
            st.error(f"No CSV files found in {data_folder}")
            return all_data, weekly_data, seasonal_data
        
        # Process each file
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            parts = file_name.replace('.csv', '').split('_')
            
            if len(parts) < 3:
                st.warning(f"Skipping file with invalid naming format: {file_name}")
                continue
            
            position = parts[0]
            year = parts[1]
            period = parts[2]
            
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Check if 'full' or 'season' in period (case insensitive)
                if 'full' in period.lower() or 'season' in period.lower():
                    if year not in seasonal_data:
                        seasonal_data[year] = {}
                    seasonal_data[year][position] = df
                else:
                    # Assume it's weekly data
                    week = period.replace('week', '').strip()
                    if year not in weekly_data:
                        weekly_data[year] = {}
                    if position not in weekly_data[year]:
                        weekly_data[year][position] = {}
                    weekly_data[year][position][week] = df
            except Exception as e:
                st.error(f"Error loading {file_name}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error accessing data folder: {str(e)}")
    
    all_data = {
        'weekly': weekly_data,
        'seasonal': seasonal_data
    }
    
    return all_data, weekly_data, seasonal_data

# Function to compute player metrics for a given year and position
def compute_player_metrics(weekly_data, seasonal_data, year, position):
    """
    Compute player metrics including volatility for a specific year and position
    """
    if year not in weekly_data or position not in weekly_data[year]:
        return pd.DataFrame()
    
    weeks = sorted(weekly_data[year][position].keys(), key=lambda x: int(x) if x.isdigit() else 0)
    if not weeks:
        return pd.DataFrame()
    
    # Get all players from the first week
    first_week_df = weekly_data[year][position][weeks[0]]
    
    # Handle column names based on the provided header information
    # Headers: Player CMP ATT PCT YDS Y/A TD INT SACKS ATT YDS TD FL G FPTS FPTS/G ROST
    player_column = 'Player'
    
    # Ensure the player column exists
    if player_column not in first_week_df.columns:
        # Try to identify the player column based on common patterns
        player_column = first_week_df.columns[0]  # Fallback to first column
    
    # Extract all unique players
    all_players = first_week_df[player_column].unique()
    
    # Prepare the metrics dataframe
    metrics_data = []
    
    for player in all_players:
        player_data = {'Player': player, 'Position': position, 'Year': year}
        game_scores = []
        
        # Collect weekly scores
        for week in weeks:
            week_df = weekly_data[year][position][week]
            player_row = week_df[week_df[player_column] == player]
            
            if not player_row.empty:
                # Use FPTS column for fantasy points
                score_column = 'FPTS'
                
                # Fallback options if FPTS doesn't exist
                if score_column not in player_row.columns:
                    score_column = next((col for col in player_row.columns if 'PTS' in col.upper()), None)
                    if not score_column:
                        score_column = next((col for col in player_row.columns if 'FP' in col.upper()), None)
                        if not score_column:
                            # If we still don't have a score column, use the fantasy points per game column
                            score_column = 'FPTS/G' if 'FPTS/G' in player_row.columns else None
                
                if score_column and score_column in player_row.columns:
                    score = player_row[score_column].iloc[0]
                    
                    if pd.notna(score) and score != '':
                        try:
                            score = float(score)
                            game_scores.append(score)
                            player_data[f'Week_{week}'] = score
                        except:
                            pass
        
        # Calculate metrics
        if game_scores:
            avg_ppg = np.mean(game_scores)
            player_data['PPG'] = avg_ppg
            player_data['Games_Played'] = len(game_scores)
            player_data['Volatility_Score'] = calculate_volatility_score(game_scores, avg_ppg)
            player_data['Consistency_Score'] = calculate_consistency_score(game_scores, avg_ppg)
            player_data['Fantasy_Value'] = calculate_player_value({
                'position': position,
                'ppg': avg_ppg,
                'game_scores': game_scores,
                'games_played': len(game_scores),
                'reliability': 0.95  # Default reliability
            })
            player_data['Value_Tier'] = assign_value_tier(player_data['Fantasy_Value'])
            
            metrics_data.append(player_data)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Add seasonal data if available
    if year in seasonal_data and position in seasonal_data[year]:
        season_df = seasonal_data[year][position]
        
        # Ensure we're using the right player column
        player_column = 'Player' if 'Player' in season_df.columns else season_df.columns[0]
        
        # Find relevant columns to merge
        # Using the provided headers: Player CMP ATT PCT YDS Y/A TD INT SACKS ATT YDS TD FL G FPTS FPTS/G ROST
        stat_columns = ['CMP', 'ATT', 'PCT', 'YDS', 'Y/A', 'TD', 'INT', 'SACKS', 
                       'ATT', 'YDS', 'TD', 'FL', 'G', 'FPTS', 'FPTS/G', 'ROST']
        
        # Get columns that exist in the dataframe
        relevant_cols = [col for col in season_df.columns if col != player_column]
        
        # Rename columns to avoid conflicts
        rename_dict = {col: f'Season_{col}' for col in relevant_cols}
        season_df_renamed = season_df.rename(columns=rename_dict)
        
        # Merge with computed metrics
        metrics_df = pd.merge(
            metrics_df, 
            season_df_renamed, 
            left_on='Player', 
            right_on=player_column, 
            how='left'
        )
        
        # Drop duplicate player column
        if player_column != 'Player' and player_column in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=[player_column])
            
        # Use season FPTS/G as reliability check if it exists
        if 'Season_FPTS/G' in metrics_df.columns:
            metrics_df['Season_PPG'] = metrics_df['Season_FPTS/G']
    
    return metrics_df

# Function to plot volatility trend
def plot_volatility_trend(player_data, player_name):
    """
    Plot the volatility trend for a specific player
    """
    # Extract week columns
    week_cols = [col for col in player_data.columns if col.startswith('Week_')]
    
    if not week_cols:
        return None
    
    # Get player row
    player_row = player_data[player_data['Player'] == player_name]
    
    if player_row.empty:
        return None
    
    # Extract scores
    weeks = [int(col.split('_')[1]) for col in week_cols]
    scores = [player_row[col].iloc[0] if col in player_row.columns and not pd.isna(player_row[col].iloc[0]) else np.nan for col in week_cols]
    
    # Filter out NaN values
    valid_indices = [i for i, score in enumerate(scores) if not pd.isna(score)]
    valid_weeks = [weeks[i] for i in valid_indices]
    valid_scores = [scores[i] for i in valid_indices]
    
    if not valid_scores:
        return None
    
    # Calculate average PPG
    avg_ppg = np.mean(valid_scores)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot weekly scores
    ax.plot(valid_weeks, valid_scores, marker='o', linestyle='-', color='#3b82f6', linewidth=2, markersize=8)
    
    # Add average line
    ax.axhline(y=avg_ppg, color='#dc2626', linestyle='--', alpha=0.7, label=f'Avg: {avg_ppg:.1f}')
    
    # Add area coloring for deviations
    for i in range(len(valid_weeks)):
        if valid_scores[i] > avg_ppg:
            # Positive deviation - green
            plt.fill_between([valid_weeks[i]-0.4, valid_weeks[i]+0.4], 
                            [avg_ppg, avg_ppg], 
                            [valid_scores[i], valid_scores[i]], 
                            color='#047857', alpha=0.3)
        else:
            # Negative deviation - red
            plt.fill_between([valid_weeks[i]-0.4, valid_weeks[i]+0.4], 
                            [avg_ppg, avg_ppg], 
                            [valid_scores[i], valid_scores[i]], 
                            color='#dc2626', alpha=0.3)
    
    # Calculate metrics
    volatility_score = calculate_volatility_score(valid_scores, avg_ppg)
    consistency_score = calculate_consistency_score(valid_scores, avg_ppg)
    
    # Customize the plot
    title = f"{player_name} Weekly Performance\nVolatility: {volatility_score:.3f} | Consistency: {consistency_score:.3f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Fantasy Points', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Set x-axis to show all weeks
    ax.set_xticks(valid_weeks)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Function to plot position comparison
def plot_position_comparison(metrics_df, position, min_games=4):
    """
    Create a scatter plot comparing volatility vs. PPG for a position
    """
    if metrics_df.empty:
        return None
    
    # Filter for the position and minimum games played
    pos_df = metrics_df[(metrics_df['Position'] == position) & (metrics_df['Games_Played'] >= min_games)]
    
    if pos_df.empty:
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define a custom colormap (Green to Red)
    cmap = LinearSegmentedColormap.from_list("volatility_cmap", ["#dc2626", "#e5e5e5", "#047857"], N=256)
    
    # Create scatter plot
    scatter = ax.scatter(
        pos_df['PPG'], 
        pos_df['Fantasy_Value'],
        c=pos_df['Volatility_Score'], 
        cmap=cmap,
        s=100 * pos_df['Consistency_Score'] + 50,  # Size based on consistency
        alpha=0.7,
        edgecolors='#1e3a8a',
        linewidths=1
    )
    
    # Add player names as annotations
    for i, row in pos_df.iterrows():
        ax.annotate(
            row['Player'],
            (row['PPG'], row['Fantasy_Value']),
            fontsize=8,
            ha='center',
            va='bottom',
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Volatility Score', fontsize=12)
    
    # Add a legend for bubble size
    for consistency in [0.25, 0.5, 0.75]:
        plt.scatter([], [], s=100 * consistency + 50, c='gray', alpha=0.7, edgecolors='#1e3a8a', linewidths=1,
                   label=f'Consistency: {consistency:.2f}')
    plt.legend(loc='upper left', title='Bubble Size Legend')
    
    # Customize the plot
    ax.set_title(f"{position} Player Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel('Points Per Game (PPG)', fontsize=12)
    ax.set_ylabel('Fantasy Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add quadrant lines and labels
    ppg_median = pos_df['PPG'].median()
    value_median = pos_df['Fantasy_Value'].median()
    
    ax.axhline(y=value_median, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=ppg_median, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(pos_df['PPG'].max(), pos_df['Fantasy_Value'].max(), 'Elite', 
            ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    ax.text(pos_df['PPG'].min(), pos_df['Fantasy_Value'].max(), 'Overachievers', 
            ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    ax.text(pos_df['PPG'].max(), pos_df['Fantasy_Value'].min(), 'Underperformers', 
            ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    ax.text(pos_df['PPG'].min(), pos_df['Fantasy_Value'].min(), 'Low Value', 
            ha='left', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    return fig

# Main dashboard layout
def main():
    # Load data
    all_data, weekly_data, seasonal_data = load_fantasy_data()
    
    if not all_data['weekly'] and not all_data['seasonal']:
        st.error("No data found. Please check your data folder structure and file naming.")
        st.info("Expected format: {position}_{year}_{week/full season}.csv in the 'fantasy_data' folder")
        
        # Show sample data format
        st.markdown("### Expected CSV Headers")
        st.code("Player, CMP, ATT, PCT, YDS, Y/A, TD, INT, SACKS, ATT, YDS, TD, FL, G, FPTS, FPTS/G, ROST, Year, Position")
        
        # Show expected folder structure
        st.markdown("### Expected Folder Structure")
        st.code("""
        fantasy_data/
        ├── QB_2022_1.csv     (QB data for 2022 week 1)
        ├── QB_2022_2.csv     (QB data for 2022 week 2)
        ├── QB_2022_season.csv (QB full season data for 2022)
        ├── RB_2022_1.csv
        └── ...
        """)
        return
    
    # Sidebar for filters
    st.sidebar.markdown('<div class="sub-header">Filters</div>', unsafe_allow_html=True)
    
    # Get available years and positions
    available_years = sorted(list(weekly_data.keys()), reverse=True)
    
    if not available_years:
        st.error("No valid yearly data found in the weekly data files.")
        return
    
    selected_year = st.sidebar.selectbox("Select Year", available_years)
    
    if selected_year in weekly_data:
        available_positions = sorted(list(weekly_data[selected_year].keys()))
        selected_position = st.sidebar.selectbox("Select Position", available_positions)
    else:
        st.error(f"No position data found for year {selected_year}")
        return
    
    # Minimum games filter
    min_games = st.sidebar.slider("Minimum Games Played", 1, 17, 4)
    
    # Compute metrics for the selected year and position
    metrics_df = compute_player_metrics(weekly_data, seasonal_data, selected_year, selected_position)
    
    if metrics_df.empty:
        st.warning(f"No data available for {selected_position} in {selected_year}")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Season View", "Weekly Breakdown"])
    
    with tab1:
        st.markdown(f'<div class="sub-header">{selected_year} {selected_position} Season Analysis</div>', unsafe_allow_html=True)
        
        # Position comparison chart
        st.subheader("Position Value Comparison")
        position_fig = plot_position_comparison(metrics_df, selected_position, min_games)
        if position_fig:
            st.pyplot(position_fig)
        else:
            st.info(f"Insufficient data for {selected_position} comparison")
        
        # Top players table
        st.subheader("Top Players by Fantasy Value")
        
        # Filter for minimum games
        filtered_df = metrics_df[metrics_df['Games_Played'] >= min_games].sort_values('Fantasy_Value', ascending=False)
        
        if not filtered_df.empty:
            # Select columns to display
            display_cols = [
                'Player', 'PPG', 'Games_Played', 'Volatility_Score', 
                'Consistency_Score', 'Fantasy_Value', 'Value_Tier'
            ]
            
            # Format the dataframe
            display_df = filtered_df[display_cols].copy()
            display_df = display_df.rename(columns={
                'PPG': 'Points/Game',
                'Games_Played': 'Games',
                'Volatility_Score': 'Volatility',
                'Consistency_Score': 'Consistency',
                'Fantasy_Value': 'Value',
                'Value_Tier': 'Tier'
            })
            
            # Format numeric columns
            display_df['Points/Game'] = display_df['Points/Game'].map('{:.1f}'.format)
            display_df['Volatility'] = display_df['Volatility'].map('{:.3f}'.format)
            display_df['Consistency'] = display_df['Consistency'].map('{:.3f}'.format)
            display_df['Value'] = display_df['Value'].map('{:.1f}'.format)
            
            # Display the table
            st.dataframe(display_df, use_container_width=True)
            
            # Advanced Stats Section
            st.subheader("Advanced Player Stats")
            
            # Display additional seasonal stats if available
            if 'Season_FPTS' in filtered_df.columns:
                # Create tabs for different stat categories
                stat_tabs = st.tabs(["Fantasy Stats", "Performance Metrics", "Raw Stats"])
                
                with stat_tabs[0]:
                    # Fantasy performance stats
                    fantasy_cols = ['Player']
                    
                    # Add fantasy-related columns
                    for col in filtered_df.columns:
                        if any(stat in col for stat in ['FPTS', 'PPG', 'G', 'ROST']):
                            if 'Season_' in col:
                                fantasy_cols.append(col)
                    
                    if len(fantasy_cols) > 1:
                        fantasy_df = filtered_df[fantasy_cols].copy()
                        st.dataframe(fantasy_df, use_container_width=True)
                    else:
                        st.info("No additional fantasy stats available")
                
                with stat_tabs[1]:
                    # Advanced metrics
                    metrics_cols = ['Player', 'Volatility_Score', 'Consistency_Score', 'Fantasy_Value']
                    metrics_df = filtered_df[metrics_cols].copy()
                    
                    # Add week-by-week volatility calculation
                    weeks_cols = [col for col in filtered_df.columns if col.startswith('Week_')]
                    if weeks_cols:
                        st.subheader("Week-by-Week Volatility")
                        
                        # Get top 5 players for detailed analysis
                        top_players = filtered_df.head(5)['Player'].tolist()
                        selected_detail_player = st.selectbox(
                            "Select player for weekly volatility breakdown",
                            options=top_players
                        )
                        
                        player_row = filtered_df[filtered_df['Player'] == selected_detail_player]
                        
                        if not player_row.empty:
                            # Extract weekly scores
                            weekly_scores = []
                            for week_col in sorted(weeks_cols, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0):
                                if pd.notna(player_row[week_col].iloc[0]):
                                    week_num = week_col.split('_')[1]
                                    score = player_row[week_col].iloc[0]
                                    weekly_scores.append({"Week": int(week_num), "Score": score})
                            
                            if weekly_scores:
                                weekly_df = pd.DataFrame(weekly_scores)
                                avg_score = weekly_df['Score'].mean()
                                
                                # Calculate weekly volatility
                                weekly_df['Deviation'] = weekly_df['Score'] - avg_score
                                weekly_df['Volatility'] = weekly_df.apply(
                                    lambda row: (row['Deviation'] / avg_score) * 0.5 if row['Deviation'] > 0 
                                    else (row['Deviation'] / avg_score) * 1.0, axis=1
                                )
                                
                                # Display the weekly breakdown
                                st.write(f"Average PPG: {avg_score:.2f}")
                                weekly_df = weekly_df.sort_values('Week')
                                
                                # Create a column chart
                                weekly_chart_data = weekly_df[['Week', 'Score', 'Volatility']]
                                st.bar_chart(weekly_chart_data.set_index('Week')['Volatility'])
                                
                                # Show the table with coloring
                                st.dataframe(weekly_df, use_container_width=True)
                    
                    st.dataframe(metrics_df, use_container_width=True)
                
                with stat_tabs[2]:
                    # Raw stats
                    raw_cols = ['Player']
                    
                    # Add raw stats columns
                    for col in filtered_df.columns:
                        if any(stat in col for stat in ['CMP', 'ATT', 'PCT', 'YDS', 'Y/A', 'TD', 'INT', 'SACKS', 'FL']):
                            if 'Season_' in col:
                                raw_cols.append(col)
                    
                    if len(raw_cols) > 1:
                        raw_df = filtered_df[raw_cols].copy()
                        st.dataframe(raw_df, use_container_width=True)
                    else:
                        st.info("No additional raw stats available")
        else:
            st.info(f"No players with at least {min_games} games")
    
    with tab2:
        st.markdown(f'<div class="sub-header">{selected_year} {selected_position} Weekly Breakdown</div>', unsafe_allow_html=True)
        
        # Player selector
        players_with_sufficient_games = metrics_df[metrics_df['Games_Played'] >= min_games]['Player'].tolist()
        
        if players_with_sufficient_games:
                            selected_player = st.selectbox(
                "Select Player", 
                sorted(players_with_sufficient_games))