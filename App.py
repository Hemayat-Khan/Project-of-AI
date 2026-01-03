import subprocess

# Install required libraries
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import calendar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output


# Set page configuration
st.set_page_config(page_title="Restaurant Management Dashboard", page_icon=":bar_chart:", layout="wide")

# Title of the web application
st.title("Restaurant Management Dashboard")

# Load the CSV file
@st.cache_data
def load_data():
    data = pd.read_csv("test_data.csv")
    return data

df = load_data()

# Extract date and hour from the 'Order_Time' column with the correct datetime format
df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%d/%m/%Y %H:%M')
df['Minute'] = df['Order_Time'].dt.minute
df['Date'] = df['Order_Time'].dt.date
df['Hour'] = df['Order_Time'].dt.hour
df['Month'] = df['Order_Time'].dt.month
df['Month_Text'] = df['Month'].apply(lambda x: calendar.month_name[x])
df['Serve_Time'] = pd.to_datetime(df['Serve_Time'], format='%M:%S.%f')
df['Serve_Time'] = df['Serve_Time'].dt.strftime('%M:%S')

# Create columns for layout
col1, col2 = st.columns(2)

################# Column 1 #################
with col1:
    # Dropdown for selecting a month
    selected_month = st.selectbox("Select a Month", df['Month_Text'].unique())
    # Filter the DataFrame based on the selected month
    filtered_df = df[df['Month_Text'] == selected_month]
    total_sale = filtered_df['Price'].sum()
    st.header(f"Total Sale for {selected_month}: ฿{total_sale}")
    # Group by 'Menu' and calculate the total sale for each item
    monthly_sale = filtered_df.groupby('Menu')['Price'].sum().reset_index()
    # Create a bar chart for monthly sales
    fig1 = px.bar(monthly_sale, x='Menu', y='Price', color='Menu', title=f'Monthly Sale for each Food/Drink in Menu ({selected_month})', labels={'x': 'Menu', 'y': 'Sale (Baht)'})
    st.plotly_chart(fig1, use_container_width=True)

################# Column 2 #################
with col2:
    # Dropdown for selecting a category
    selected_Category = st.selectbox("Select a Category", df['Category'].unique())
    # Filter the DataFrame based on the selected category
    filtered_df = df[df['Category'] == selected_Category]
    # Calculate total sale for the selected category
    total_sale = filtered_df['Price'].sum()
    st.header(f"Total Sale of {selected_Category}: ฿{total_sale}")
    # Group by day of the week and calculate the count of orders for each day
    daily_sales = filtered_df.groupby('Day_Of_Week')['Category'].count().reset_index()
    # Define the order of days of the week
    custom_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Set the 'Day_Of_Week' column as a category with custom order
    daily_sales['Day_Of_Week'] = pd.Categorical(daily_sales['Day_Of_Week'], categories=custom_order, ordered=True)
    # Sort the DataFrame based on the custom order
    daily_sales = daily_sales.sort_values('Day_Of_Week')
    # Create a bar chart for daily sales
    fig2 = px.bar(daily_sales, x='Day_Of_Week', y='Category', color='Day_Of_Week',  
              title=f'Daily Sales for {selected_Category} by Day of the Week', labels={'Day_Of_Week': 'Day of the Week', 'Category': 'Total Sales (Baht)'})
    st.plotly_chart(fig2, use_container_width=True)

# Create columns for layout
col3, col4 = st.columns(2)

################# Column 3 #################
with col3:
    # Create a pivot table for minimum serve time for each drinks staff
    pivot_df = pd.pivot_table(df, values='Serve_Time', index='Menu', columns='Drinks_Staff', aggfunc='min')
    # Reset the index to have 'Menu' as a regular column
    pivot_df.reset_index(inplace=True)
    # Melt the DataFrame to have a column 'Drinks_Staff' with values 1, 2, 3
    melted_df = pd.melt(pivot_df, id_vars=['Menu'], var_name='Drinks_Staff', value_name='Minimum_Serve_Time')
    melted_df = melted_df.sort_values(by=['Drinks_Staff', 'Minimum_Serve_Time'])
    # Create a grouped bar plot for minimum serve time for each drinks staff
    fig3 = px.bar(melted_df, x='Menu', y='Minimum_Serve_Time', color='Drinks_Staff', barmode='group', 
              title='Minimum Serve Time for each Drinks Staff in Menu', labels={'Drinks_Staff': 'Number of Staff', 'Minimum_Serve_Time': 'Minimum Serve Time (Minute)'})
    st.plotly_chart(fig3, use_container_width=True)

################# Column 4 #################
with col4:
    # Create a pivot table for minimum serve time for each kitchen staff
    pivot_df = pd.pivot_table(df, values='Serve_Time', index='Menu', columns='Kitchen_Staff', aggfunc='min')
    # Reset the index to have 'Menu' as a regular column
    pivot_df.reset_index(inplace=True)
    # Melt the DataFrame to have a column 'Kitchen_Staff' with values 1, 2, 3
    melted_df = pd.melt(pivot_df, id_vars=['Menu'], var_name='Kitchen_Staff', value_name='Minimum_Serve_Time')
    melted_df = melted_df.sort_values(by=['Kitchen_Staff', 'Minimum_Serve_Time'])
    # Create a grouped bar plot for minimum serve time for each kitchen staff
    fig4 = px.bar(melted_df, x='Menu', y='Minimum_Serve_Time', color='Kitchen_Staff', barmode='group', 
              title='Minimum Serve Time for each Kitchen Staff in Menu', labels={'Kitchen_Staff': 'Number of Staff', 'Minimum_Serve_Time': 'Minimum Serve Time (Minute)'})
    st.plotly_chart(fig4, use_container_width=True)

# Create columns for layout
col5, col6 = st.columns(2)

################# Column 5 #################
with col5:
    # Get the top 5 most sold food items
    top_food = df[df['Category'] == 'food'].groupby('Menu')['Price'].sum().nlargest(5).reset_index()
    top_food['Category'] = 'Food'  # Add a new column indicating the category
    # Get the top 5 most sold drink items
    top_drink = df[df['Category'] == 'drink'].groupby('Menu')['Price'].sum().nlargest(5).reset_index()
    top_drink['Category'] = 'Drink'  # Add a new column indicating the category
    # Combine data for both food and drink
    top_items = pd.concat([top_food, top_drink])
    # Create a bar chart for the top 5 most sold items with color-coded categories
    fig_top_items = px.bar(top_items, x='Menu', y='Price', color='Category', title='Top 5 Most Sold Items', labels={'x': 'Menu', 'y': 'Sale', 'Category': 'Category'})
    # Display the top 5 most sold items
    st.plotly_chart(fig_top_items, use_container_width=True)

################# Column 6 #################
with col6:
    # Create a subplot for category distribution of drink and food
    fig6 = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]], subplot_titles=['Category Distribution - Drink', 'Category Distribution - Food'])
    # Group by 'Category' and 'Menu' to get the count
    category_menu_count = df.groupby(['Category', 'Menu']).size().reset_index(name='Count')
    # Filter data for 'drink'
    drink_data = category_menu_count[category_menu_count['Category'] == 'drink']
    # Plot Pie Chart for 'drink'
    fig6.add_trace(go.Pie(labels=drink_data['Menu'], values=drink_data['Count'], name='Drink', hoverinfo='label+percent+name'), row=1, col=1)
    # Filter data for 'food'
    food_data = category_menu_count[category_menu_count['Category'] == 'food']
    # Plot Pie Chart for 'food'
    fig6.add_trace(go.Pie(labels=food_data['Menu'], values=food_data['Count'], name='Food', hoverinfo='label+percent+name'), row=1, col=2)
    # Update layout
    fig6.update_layout(title_text='Category Distribution - Drink and Food', showlegend=True)
    # Display the subplot
    st.plotly_chart(fig6, use_container_width=True)
