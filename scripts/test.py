import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Simple Fantasy Football Dashboard")

# Get list of files in fantasy_data folder
files = os.listdir("fantasy_data")
csv_files = [f for f in files if f.endswith('.csv')]

# Let user select a file to display
if csv_files:
    selected_file = st.selectbox("Select a file to display:", csv_files)
    
    # Read the selected file
    df = pd.read_csv(os.path.join("fantasy_data", selected_file))
    
    # Display the dataframe
    st.write(f"### Data from {selected_file}")
    st.dataframe(df)
    
    # Check if it has fantasy points column
    if 'FPTS' in df.columns:
        # Show top players by fantasy points
        st.write("### Top Players by Fantasy Points")
        top_players = df.sort_values('FPTS', ascending=False).head(10)
        
        # Create a simple bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='FPTS', y='Player', data=top_players, ax=ax)
        ax.set_title("Top Players by Fantasy Points")
        st.pyplot(fig)
        
        # Calculate simple statistics
        st.write("### Fantasy Points Statistics")
        st.write(f"Average FPTS: {df['FPTS'].mean():.2f}")
        st.write(f"Median FPTS: {df['FPTS'].median():.2f}")
        st.write(f"Max FPTS: {df['FPTS'].max():.2f}")
        st.write(f"Min FPTS: {df['FPTS'].min():.2f}")
    else:
        st.warning("This file doesn't have a FPTS column.")
else:
    st.error("No CSV files found in the fantasy_data folder")