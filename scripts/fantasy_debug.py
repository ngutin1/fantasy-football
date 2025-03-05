import streamlit as st
import pandas as pd
import os
import glob

st.title("Fantasy Football Data Debugger")

# Check if fantasy_data folder exists
if os.path.exists("fantasy_data"):
    st.success("✅ fantasy_data folder exists")
    
    # List all files in the folder
    files = os.listdir("fantasy_data")
    st.write(f"Found {len(files)} files in fantasy_data folder:")
    
    for file in files:
        if file.endswith(".csv"):
            st.write(f"📄 {file}")
            
            # Try to read the file
            try:
                file_path = os.path.join("fantasy_data", file)
                df = pd.read_csv(file_path)
                st.write(f"  ✅ Successfully read file with {len(df)} rows and {len(df.columns)} columns")
                
                # Show first few columns
                st.write(f"  Column names: {', '.join(df.columns[:5])}...")
                
                # Show first few rows
                st.write("  Sample data:")
                st.dataframe(df.head(3))
                
                st.write("---")
            except Exception as e:
                st.error(f"  ❌ Error reading file: {str(e)}")
else:
    st.error("❌ fantasy_data folder not found")
    
    # Show current working directory
    st.write(f"Current working directory: {os.getcwd()}")
    
    # Suggest creating the folder
    if st.button("Create fantasy_data folder"):
        os.makedirs("fantasy_data")
        st.success("Created fantasy_data folder. Please add your CSV files to it.")