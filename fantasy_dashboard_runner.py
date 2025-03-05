# Run this file to start the Streamlit dashboard
# Save the dashboard code to a file first

import os
import streamlit as st

# Create the necessary directory structure if it doesn't exist
def setup_directory():
    # Create the fantasy_data directory if it doesn't exist
    if not os.path.exists('fantasy_data'):
        os.makedirs('fantasy_data')
        print("Created 'fantasy_data' directory")
    
    # Create a sample data file for testing if no files exist
    if not any(file.endswith('.csv') for file in os.listdir('fantasy_data')):
        create_sample_data()

# Create sample data files for testing
def create_sample_data():
    # Sample QB data for week 1
    qb_week1 = """Player,CMP,ATT,PCT,YDS,Y/A,TD,INT,SACKS,ATT,YDS,TD,FL,G,FPTS,FPTS/G,ROST
Patrick Mahomes,30,39,76.9,360,9.2,5,0,2,3,25,0,0,1,41.9,41.9,100
Josh Allen,26,33,78.8,297,9.0,3,2,1,10,56,1,0,1,35.5,35.5,100
Jalen Hurts,22,33,66.7,279,8.5,3,0,2,15,90,1,0,1,34.2,34.2,100
Lamar Jackson,17,22,77.3,213,9.7,3,0,1,11,95,1,0,1,33.0,33.0,100
Joe Burrow,23,31,74.2,259,8.4,3,1,2,5,15,0,0,1,23.9,23.9,99"""
    
    # Sample QB data for week 2
    qb_week2 = """Player,CMP,ATT,PCT,YDS,Y/A,TD,INT,SACKS,ATT,YDS,TD,FL,G,FPTS,FPTS/G,ROST
Patrick Mahomes,33,44,75.0,343,7.8,3,1,0,2,15,0,0,1,27.2,27.2,100
Josh Allen,18,27,66.7,218,8.1,4,0,1,6,34,0,0,1,30.1,30.1,100
Jalen Hurts,20,29,69.0,193,6.7,1,1,3,11,61,2,0,1,24.8,24.8,100
Lamar Jackson,18,29,62.1,187,6.4,1,0,2,9,119,1,0,1,24.6,24.6,100
Joe Burrow,29,42,69.0,302,7.2,3,2,6,4,26,0,0,1,24.7,24.7,99"""
    
    # Sample QB season data
    qb_season = """Player,CMP,ATT,PCT,YDS,Y/A,TD,INT,SACKS,ATT,YDS,TD,FL,G,FPTS,FPTS/G,ROST
Patrick Mahomes,382,584,65.4,4183,7.2,37,9,23,78,410,5,2,16,345.3,21.6,100
Josh Allen,359,567,63.3,4306,7.6,35,18,33,124,762,7,6,17,361.4,21.3,100
Jalen Hurts,265,388,68.3,3233,8.3,22,6,39,165,760,15,5,15,352.9,23.5,100
Lamar Jackson,274,426,64.3,3678,8.6,27,12,26,152,821,5,3,15,329.2,21.9,100
Joe Burrow,414,606,68.3,4260,7.0,31,12,51,75,257,5,5,16,317.1,19.8,99"""
    
    # Write files
    with open('fantasy_data/QB_2023_1.csv', 'w') as f:
        f.write(qb_week1)
    
    with open('fantasy_data/QB_2023_2.csv', 'w') as f:
        f.write(qb_week2)
    
    with open('fantasy_data/QB_2023_season.csv', 'w') as f:
        f.write(qb_season)
    
    print("Created sample data files in 'fantasy_data' directory")

# Run the dashboard
def run_dashboard():
    # Ensure the dashboard file exists
    dashboard_file = 'fantasy_dashboard.py'
    
    if not os.path.exists(dashboard_file):
        print(f"Error: {dashboard_file} not found. Please save the dashboard code first.")
        return
    
    # Run the Streamlit dashboard
    os.system(f"streamlit run {dashboard_file}")

if __name__ == "__main__":
    setup_directory()
    run_dashboard()
    print("Dashboard is running. Open your browser to view it.")

