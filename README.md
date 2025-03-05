# Fantasy Football Volatility Dashboard

This repository contains two different versions of a fantasy football volatility dashboard designed to help fantasy managers analyze player performance volatility:

1. **Standard Volatility Dashboard** - Emphasizes consistency by penalizing underperformance
2. **Progressive Volatility Dashboard** - Uses quadratic scaling for both over and underperformance

Both dashboards were designed for SuperFlex, 3WR, PPR, TE Premium league settings but can be used for any fantasy football format.

## Features

- Player volatility calculations and comparisons
- Customizable by year and position group
- Visual representation of weekly performance vs average
- Detailed volatility contribution breakdown
- Color-coded performance indicators
- Minimum games played filter

## Volatility Metrics Explained

### Standard Volatility Metric (Dashboard 1)

This metric gives more weight to underperformance, which helps identify consistent, reliable performers:

```python
def calculate_volatility_score(game_scores, average_ppg):
    """
    Volatility score with heavier penalty for underperformance
    """
    volatility_score = 0
    total_games = len(game_scores)
    
    for score in game_scores:
        deviation = score - average_ppg
        
        if deviation > 0:
            # Positive deviation (exceeded average)
            volatility_score += (deviation / average_ppg) * 0.5  # 50% weight for exceeding
        else:
            # Negative deviation (below average)
            volatility_score += (deviation / average_ppg) * 1.0  # 100% weight for underperforming
    
    return volatility_score / total_games
```

### Progressive Volatility Metric (Dashboard 2)

This metric uses quadratic scaling for both over and underperformance, emphasizing exceptional performances:

```python
def calculate_volatility_score(game_scores, average_ppg):
    """
    Volatility score with progressive scaling for both over and underperformance
    """
    volatility_score = 0
    total_games = len(game_scores)
    
    for score in game_scores:
        deviation = score - average_ppg
        relative_deviation = deviation / average_ppg
        
        if deviation > 0:
            # Progressive scaling for overperformance
            volatility_score += relative_deviation * (1 + relative_deviation)
        else:
            # Progressive scaling for underperformance
            volatility_score += relative_deviation * (1 - relative_deviation)
    
    return volatility_score / total_games
```

## Installation and Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/fantasy-football-volatility.git
```

2. Install the required packages:
```
pip install streamlit pandas numpy matplotlib seaborn
```

3. Place your fantasy football data files in a folder named `fantasy_data` with the following naming convention:
```
{position}_{year}_{week/full}_{#/season}.csv
```
Examples:
- `qb_2023_week_1.csv`
- `rb_2023_full_season.csv`

4. Run either dashboard:
```
streamlit run standard_volatility_dashboard.py
```
or
```
streamlit run progressive_volatility_dashboard.py
```

## Data Format

Your CSV files should contain the following columns:
- `Player` - Player name (or Team if columns are swapped)
- `Team` - Team name (or Player if columns are swapped)
- Fantasy points columns (one of): `FPTS`, `PTS`, or similar

The dashboard will automatically detect and handle column swaps between Player and Team.

## Interpreting Volatility Scores

### Standard Volatility
- **Positive scores**: Player tends to exceed their average
- **Scores near zero**: Player performs consistently around their average
- **Negative scores**: Player tends to underperform their average

### Progressive Volatility
- **High positive scores (>0.5)**: Player has exceptional upside games
- **Moderate positive scores (0.1-0.5)**: Player has good upside with limited downside
- **Scores near zero (-0.1 to 0.1)**: Player performs consistently around their average
- **Moderate negative scores (-0.5 to -0.1)**: Player has concerning underperforming games
- **High negative scores (<-0.5)**: Player has extreme underperforming games

## Dashboard Files

1. `standard_volatility_dashboard.py` - Version with higher penalty for underperformance
2. `progressive_volatility_dashboard.py` - Version with quadratic scaling in both directions

## Use Cases

- **Standard Volatility**: Better for head-to-head leagues where consistency is key
- **Progressive Volatility**: Better for tournament-style contests and best ball formats where ceiling games matter more

## License

This project is licensed under the MIT License - see the LICENSE file for details.
