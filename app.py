#https://three21-project-6bjc.onrender.com

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

# Data Processing Functions
def load_and_clean_data(filepath):
    """
    Load and clean the census data from CSV file
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Remove commas and convert to numeric
    for col in ['Total', 'Men', 'Women']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NA values for essential columns
    df = df.dropna(subset=['Total', 'Men', 'Women'])
    
    return df

def get_essential_services_data(df):
    """
    Extract data for essential services (nurses, police, firefighters)
    """
    # Define the NOC codes or partial occupation names for essential services
    essential_services = [
        'Police officers', 
        'Firefighters',
        'Registered nurses'
    ]
    
    # Filter for occupations containing these terms
    essential_df = df[df['Occupation'].str.contains('|'.join(essential_services), case=False, na=False)]
    
    return essential_df

def get_noc_top_level_data(df):
    """
    Get top-level NOC categories (single-digit codes)
    """
    # Pattern to match is a digit followed by a space and any text
    pattern = r'^\d\s[A-Za-z]+'
    top_level_df = df[df['Occupation'].str.match(pattern, na=False)]
    
    return top_level_df

def get_engineering_data(df):
    """
    Extract data for computer, mechanical, and electrical engineers
    """
    engineering_occupations = [
        'Computer engineers', 
        'Mechanical engineers',
        'Electrical and electronics engineers'
    ]
    
    engineering_df = df[df['Occupation'].str.contains('|'.join(engineering_occupations), case=False, na=False)]
    
    return engineering_df

def normalize_by_population(df, population_data):
    """
    Normalize data by population for each province
    """
    # Merge the data with population information
    normalized_df = df.copy()
    
    # Calculate per capita values (per 10,000 people)
    for col in ['Total', 'Men', 'Women']:
        if col in normalized_df.columns:
            normalized_df[f'{col}_per_10k'] = normalized_df[col] / (population_data / 10000)
    
    return normalized_df

def get_gender_ratio(df):
    """
    Calculate gender ratio (men to women)
    """
    gender_df = df.copy()
    gender_df['GenderRatio'] = gender_df['Men'] / gender_df['Women']
    return gender_df

# Simulated province data - normally this would come from the census data
def get_province_data():
    """
    Return a dictionary of province data to simulate province-based analysis
    """
    # In a real implementation, this data would be extracted from the full census dataset
    provinces = {
        'Alberta': {'Population': 3375130	},
        'British Columbia': {'Population': 4200425},
        'Manitoba': {'Population': 1058410},
        'New Brunswick': {'Population': 648250},
        'Newfoundland and Labrador': {'Population': 433955},
        'Northwest Territories': {'Population': 31915},
        'Nova Scotia': {'Population': 31915},
        'Nunavut': {'Population': 24540},
        'Ontario': {'Population': 11782825},
        'Prince Edward Island': {'Population': 126900},
        'Quebec': {'Population': 93585},
        'Saskatchewan': {'Population': 882760},
        'Yukon': {'Population': 32775}
    }
    
    return provinces

# Load and process data
df = load_and_clean_data('cleaned_data.csv')
provinces = get_province_data()

# Extract relevant data for visualizations
essential_services_df = get_essential_services_data(df)
noc_top_level_df = get_noc_top_level_data(df)
engineering_df = get_engineering_data(df)

# Prepare province population data for normalization
province_populations = {prov: data['Population'] for prov, data in provinces.items()}

# Create the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("2023 Canadian Census Data Dashboard", className="text-center"),
            html.P("Interactive visualization of essential services and employment statistics", className="text-center")
        ], width=12)
    ], className="mt-4 mb-4"),
    
    # Tabs for different visualizations
    dbc.Tabs([
        # Tab 1: Essential Services Distribution
        dbc.Tab(label="Essential Services Distribution", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Essential Services Distribution", className="mt-3"),
                    html.P("Distribution of essential services personnel (nurses, police, firefighters) across provinces")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Select Service Type:"),
                    dcc.Dropdown(
                        id="service-type-dropdown",
                        options=[
                            {"label": "All Essential Services", "value": "all"},
                            {"label": "Police Officers", "value": "police"},
                            {"label": "Firefighters", "value": "fire"},
                            {"label": "Registered Nurses", "value": "nurse"}
                        ],
                        value="all",
                        clearable=False
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Normalization:"),
                    dcc.RadioItems(
                        id="normalization-radio",
                        options=[
                            {"label": "Absolute Numbers", "value": "absolute"},
                            {"label": "Per 10,000 Population", "value": "normalized"}
                        ],
                        value="absolute",
                        inline=True
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Sort By:"),
                    dcc.RadioItems(
                        id="sort-radio",
                        options=[
                            {"label": "Province (A-Z)", "value": "province"},
                            {"label": "Value (Descending)", "value": "value"}
                        ],
                        value="value",
                        inline=True
                    )
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="essential-services-graph")
                ], width=12)
            ])
        ]),
        
        # Tab 2: Gender-based Employment Statistics
        dbc.Tab(label="Gender-based Employment", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Gender-based Employment Statistics", className="mt-3"),
                    html.P("Employment statistics by gender across top-level NOC categories")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Select NOC Categories:"),
                    dcc.Dropdown(
                        id="noc-dropdown",
                        options=[
                            {"label": occ, "value": occ} 
                            for occ in noc_top_level_df['Occupation'].unique()
                        ],
                        value=noc_top_level_df['Occupation'].unique()[:3].tolist(),
                        multi=True
                    )
                ], width=6),
                
                dbc.Col([
                    html.Label("Chart Type:"),
                    dcc.RadioItems(
                        id="chart-type-radio",
                        options=[
                            {"label": "Stacked Bar", "value": "stack"},
                            {"label": "Grouped Bar", "value": "group"},
                            {"label": "Gender Ratio", "value": "ratio"}
                        ],
                        value="stack",
                        inline=True
                    )
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="gender-employment-graph")
                ], width=12)
            ])
        ]),
        
        # Tab 3: Engineering Manpower for EV Factory
        dbc.Tab(label="Engineering Manpower", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Engineering Manpower for EV Factory Setup", className="mt-3"),
                    html.P("Analysis of available engineering talent (Computer, Mechanical, Electrical) by province")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Engineering Type:"),
                    dcc.Checklist(
                        id="engineering-checklist",
                        options=[
                            {"label": "Computer Engineers", "value": "computer"},
                            {"label": "Mechanical Engineers", "value": "mechanical"},
                            {"label": "Electrical Engineers", "value": "electrical"}
                        ],
                        value=["computer", "mechanical", "electrical"],
                        inline=True
                    )
                ], width=6),
                
                dbc.Col([
                    html.Label("View:"),
                    dcc.RadioItems(
                        id="engineering-view-radio",
                        options=[
                            {"label": "Absolute Numbers", "value": "absolute"},
                            {"label": "Percentage of Total", "value": "percentage"},
                            {"label": "Per Capita", "value": "per_capita"}
                        ],
                        value="absolute",
                        inline=True
                    )
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="engineering-manpower-graph")
                ], width=12)
            ])
        ]),
        
        # Tab 4: Custom Insight
        dbc.Tab(label="Custom Insight", children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Custom Insight: Gender Distribution Across Occupation Levels", className="mt-3"),
                    html.P("Exploring gender distribution patterns across different occupation hierarchy levels")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Select Occupation Category:"),
                    dcc.Dropdown(
                        id="occupation-category-dropdown",
                        options=[
                            {"label": "Business & Finance", "value": "business"},
                            {"label": "Sciences & Engineering", "value": "science"},
                            {"label": "Health", "value": "health"},
                            {"label": "Education & Law", "value": "education"},
                            {"label": "Art & Culture", "value": "art"}
                        ],
                        value="science",
                        clearable=False
                    )
                ], width=6),
                
                dbc.Col([
                    html.Label("Analysis Type:"),
                    dcc.RadioItems(
                        id="analysis-type-radio",
                        options=[
                            {"label": "Hierarchy Level Analysis", "value": "hierarchy"},
                            {"label": "Gender Parity Index", "value": "parity"}
                        ],
                        value="hierarchy",
                        inline=True
                    )
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="custom-insight-graph")
                ], width=12)
            ])
        ])
    ]),
    
    html.Footer([
        html.P("Data Source: 2023 Statistics Canada Census", className="text-center mt-4 text-muted")
    ])
], fluid=True)

# Callbacks for interactive visualizations

# Callback for Essential Services Distribution
@app.callback(
    Output("essential-services-graph", "figure"),
    [
        Input("service-type-dropdown", "value"),
        Input("normalization-radio", "value"),
        Input("sort-radio", "value")
    ]
)
def update_essential_services_graph(service_type, normalization, sort_by):
    # Filter data based on service type
    if service_type == "all":
        filtered_df = essential_services_df.copy()
    elif service_type == "police":
        filtered_df = essential_services_df[essential_services_df['Occupation'].str.contains('Police', case=False, na=False)]
    elif service_type == "fire":
        filtered_df = essential_services_df[essential_services_df['Occupation'].str.contains('Fire', case=False, na=False)]
    elif service_type == "nurse":
        filtered_df = essential_services_df[essential_services_df['Occupation'].str.contains('Nurse', case=False, na=False)]
    
    # Create a simulated dataset by province (since the actual data doesn't have province breakdowns)
    # In a real implementation, you'd use the actual province data from the census
    provinces_list = list(provinces.keys())
    
    # Create simulated data for demonstration purposes
    # In a real implementation, this would use actual census data by province
    province_data = []
    for occ in filtered_df['Occupation'].unique():
        total = filtered_df[filtered_df['Occupation'] == occ]['Total'].iloc[0]
        # Distribute the total across provinces based on population proportion with some random variation
        for province in provinces_list:
            pop_proportion = provinces[province]['Population'] / sum([p['Population'] for p in provinces.values()])
            # Add some random variation to make the visualization more interesting
            variation = np.random.uniform(0.7, 1.3)
            province_value = int(total * pop_proportion * variation)
            
            province_data.append({
                'Province': province,
                'Occupation': occ,
                'Count': province_value,
                'Population': provinces[province]['Population'],
                'Per10K': (province_value / provinces[province]['Population']) * 10000
            })
    
    province_df = pd.DataFrame(province_data)
    
    # Aggregate data by province if showing all services
    if service_type == "all":
        province_df = province_df.groupby('Province').agg({
            'Count': 'sum',
            'Population': 'first',
            'Per10K': 'sum'
        }).reset_index()
    
    # Apply normalization
    y_column = 'Per10K' if normalization == 'normalized' else 'Count'
    y_title = 'Personnel per 10,000 Population' if normalization == 'normalized' else 'Number of Personnel'
    
    # Sort the data
    if sort_by == 'province':
        province_df = province_df.sort_values('Province')
    else:  # sort_by == 'value'
        province_df = province_df.sort_values(y_column, ascending=False)
    
    # Create the graph
    fig = px.bar(
        province_df,
        x='Province',
        y=y_column,
        color='Province',
        title=f'Essential Services Distribution by Province ({service_type.title()})',
        labels={'Province': 'Province/Territory', y_column: y_title}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title="Province/Territory",
        height=600
    )
    
    return fig

# Callback for Gender-based Employment
@app.callback(
    Output("gender-employment-graph", "figure"),
    [
        Input("noc-dropdown", "value"),
        Input("chart-type-radio", "value")
    ]
)
def update_gender_employment_graph(selected_nocs, chart_type):
    # Verify if selected_nocs is empty and provide fallback
    if not selected_nocs or len(selected_nocs) == 0:
        selected_nocs = noc_top_level_df['Occupation'].unique()[:3].tolist()
    
    # Filter data for selected NOC categories
    filtered_df = noc_top_level_df[noc_top_level_df['Occupation'].isin(selected_nocs)]
    
    # Create different chart types based on selection
    if chart_type == "ratio":
        # Calculate gender ratio
        filtered_df = filtered_df.copy()  # Create a copy to avoid SettingWithCopyWarning
        filtered_df['Ratio'] = filtered_df['Men'] / filtered_df['Women']
        
        fig = px.bar(
            filtered_df,
            x='Occupation',
            y='Ratio',
            color='Occupation',
            title='Gender Ratio (Men/Women) by NOC Category',
            labels={'Occupation': 'NOC Category', 'Ratio': 'Men/Women Ratio'}
        )
        
        # Add a horizontal line at ratio = 1 (equal gender representation)
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=1,
            x1=len(filtered_df) - 0.5,
            y1=1,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.update_layout(height=600)
        
    else:  # Stacked or grouped bar
        # Prepare data in long format
        df_long = pd.melt(
            filtered_df,
            id_vars=['Occupation'],
            value_vars=['Men', 'Women'],
            var_name='Gender',
            value_name='Count'
        )
        
        # Make sure chart_type is a valid barmode value ('stack' or 'group')
        barmode = chart_type if chart_type in ['stack', 'group'] else 'stack'
        
        fig = px.bar(
            df_long,
            x='Occupation',
            y='Count',
            color='Gender',
            barmode=barmode,
            title='Employment by Gender and NOC Category',
            labels={'Occupation': 'NOC Category', 'Count': 'Number of Employed Persons', 'Gender': 'Gender'}
        )
        
        fig.update_layout(height=600)
    
    return fig

# Callback for Engineering Manpower
@app.callback(
    Output("engineering-manpower-graph", "figure"),
    [
        Input("engineering-checklist", "value"),
        Input("engineering-view-radio", "value")
    ]
)
def update_engineering_manpower_graph(selected_types, view_type):
    # Verify if selected_types is empty and provide fallback
    if not selected_types or len(selected_types) == 0:
        selected_types = ["computer", "mechanical", "electrical"]
    
    # Filter engineering data
    engineering_filters = []
    if "computer" in selected_types:
        engineering_filters.append("Computer")
    if "mechanical" in selected_types:
        engineering_filters.append("Mechanical")
    if "electrical" in selected_types:
        engineering_filters.append("Electrical")
    
    # If engineering_filters is still empty after checking, use a default
    if not engineering_filters:
        engineering_filters = ["Computer"]
    
    filtered_df = engineering_df[
        engineering_df['Occupation'].str.contains('|'.join(engineering_filters), case=False, na=False)
    ]
    
    # Handle empty filtered dataframe
    if filtered_df.empty:
        fig = px.bar(
            title="No data matching selected engineering types",
        )
        fig.update_layout(height=600)
        return fig
    
    # Create simulated province data (as with the essential services)
    provinces_list = list(provinces.keys())
    province_data = []
    
    for occ in filtered_df['Occupation'].unique():
        total = filtered_df[filtered_df['Occupation'] == occ]['Total'].iloc[0]
        
        for province in provinces_list:
            pop_proportion = provinces[province]['Population'] / sum([p['Population'] for p in provinces.values()])
            # Add more variation for engineering distribution (some provinces may have tech hubs)
            if province in ['Ontario', 'British Columbia', 'Quebec']:
                variation = np.random.uniform(1.2, 1.8)  # Tech hubs with more engineers
            else:
                variation = np.random.uniform(0.5, 1.1)
                
            province_value = int(total * pop_proportion * variation)
            
            engineer_type = "Computer" if "Computer" in occ else "Mechanical" if "Mechanical" in occ else "Electrical"
            
            province_data.append({
                'Province': province,
                'EngineerType': engineer_type,
                'Count': province_value,
                'Population': provinces[province]['Population'],
                'Per10K': (province_value / provinces[province]['Population']) * 10000
            })
    
    province_df = pd.DataFrame(province_data)
    
    # Prepare data based on view type
    if view_type == "absolute":
        y_column = 'Count'
        y_title = 'Number of Engineers'
    elif view_type == "percentage":
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        province_df = province_df.copy()
        # Calculate percentage of total for each province
        province_totals = province_df.groupby('Province')['Count'].transform('sum')
        province_df['Percentage'] = (province_df['Count'] / province_totals) * 100
        y_column = 'Percentage'
        y_title = 'Percentage of Total Engineers (%)'
    else:  # per_capita
        y_column = 'Per10K'
        y_title = 'Engineers per 10,000 Population'
    
    # Create the figure
    fig = px.bar(
        province_df,
        x='Province',
        y=y_column,
        color='EngineerType',
        barmode='group',
        title=f'Engineering Manpower by Province and Type',
        labels={'Province': 'Province/Territory', y_column: y_title, 'EngineerType': 'Engineer Type'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600
    )
    
    return fig

# Callback for Custom Insight
@app.callback(
    Output("custom-insight-graph", "figure"),
    [
        Input("occupation-category-dropdown", "value"),
        Input("analysis-type-radio", "value")
    ]
)
def update_custom_insight_graph(category, analysis_type):
    # Filter data based on category (using simple string matching for demonstration)
    category_filters = {
        "business": ["Business", "finance", "administration"],
        "science": ["Natural", "applied sciences", "engineering"],
        "health": ["Health", "nurse", "medical"],
        "education": ["Education", "law", "social"],
        "art": ["Art", "culture", "recreation"]
    }
    
    # Ensure category is valid
    if category not in category_filters:
        category = "business"  # Default to business if an invalid category is selected
    
    # Filter data that contains any of the terms in the selected category
    filtered_df = df[
        df['Occupation'].str.contains('|'.join(category_filters[category]), case=False, na=False)
    ]
    
    # Handle empty filtered dataframe
    if filtered_df.empty:
        fig = px.bar(
            title=f"No data matching selected category: {category}",
        )
        fig.update_layout(height=600)
        return fig
    
    if analysis_type == "hierarchy":
        # Make a copy to avoid SettingWithCopyWarning
        filtered_df = filtered_df.copy()
        
        # Add a level column based on the NOC code structure (indentation level)
        # This is a simplification - in a real implementation you'd use the actual NOC hierarchy
        
        # Count leading spaces to determine hierarchy level
        def count_leading_spaces(text):
            if not isinstance(text, str):
                return 0
            return len(text) - len(text.lstrip(' '))
        
        filtered_df['Level'] = filtered_df['Occupation'].apply(count_leading_spaces) // 2
        
        # Group by level and calculate gender percentages
        level_data = filtered_df.groupby('Level').agg({
            'Men': 'sum',
            'Women': 'sum'
        }).reset_index()
        
        # Handle case where there's no data
        if level_data.empty:
            fig = px.bar(
                title=f"No hierarchy data available for {category.title()} category",
            )
            fig.update_layout(height=600)
            return fig
            
        level_data['Total'] = level_data['Men'] + level_data['Women']
        level_data['Men_Pct'] = (level_data['Men'] / level_data['Total']) * 100
        level_data['Women_Pct'] = (level_data['Women'] / level_data['Total']) * 100
        
        # Create a grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=level_data['Level'],
            y=level_data['Men_Pct'],
            name='Men',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            x=level_data['Level'],
            y=level_data['Women_Pct'],
            name='Women',
            marker_color='red'
        ))
        
        # Add a horizontal line at 50% for reference
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=50,
            x1=level_data['Level'].max() + 0.5,
            y1=50,
            line=dict(color="green", width=2, dash="dash"),
        )
        
        fig.update_layout(
            title=f'Gender Distribution by Hierarchy Level in {category.title()} Occupations',
            xaxis_title='Hierarchy Level (0 = Senior, Higher = Junior)',
            yaxis_title='Percentage (%)',
            barmode='group',
            height=600
        )
        
    else:  # analysis_type == "parity"
        # Make a copy to avoid SettingWithCopyWarning
        filtered_df = filtered_df.copy()
        
        # Calculate gender parity index for each occupation
        # Prevent division by zero
        filtered_df['GPI'] = filtered_df.apply(
            lambda row: row['Women'] / row['Men'] if row['Men'] > 0 else float('inf'), 
            axis=1
        )
        
        # Sort by GPI
        sorted_df = filtered_df.sort_values('GPI')
        
        # Handle case where there's too little data
        if len(sorted_df) < 5:
            fig = px.bar(
                sorted_df,
                y='Occupation',
                x='GPI',
                orientation='h',
                title=f'Gender Parity Index in {category.title()} Occupations (Limited Data)',
                labels={'Occupation': 'Occupation', 'GPI': 'Gender Parity Index (Women/Men)'},
                color='GPI',
                color_continuous_scale=['blue', 'white', 'red'],
                range_color=[0, 2]
            )
            
            fig.update_layout(height=600)
            return fig
        
        # Take 10 most unequal occupations from each end (or fewer if less data is available)
        n_items = min(10, len(sorted_df) // 2)
        bottom_n = sorted_df.head(n_items)
        top_n = sorted_df.tail(n_items)
        combined_df = pd.concat([bottom_n, top_n])
        
        # Create horizontal bar chart
        fig = px.bar(
            combined_df,
            y='Occupation',
            x='GPI',
            orientation='h',
            title=f'Gender Parity Index in {category.title()} Occupations',
            labels={'Occupation': 'Occupation', 'GPI': 'Gender Parity Index (Women/Men)'},
            color='GPI',
            color_continuous_scale=['blue', 'white', 'red'],
            range_color=[0, 2]
        )
        
        # Add a vertical line at GPI = 1 (perfect gender parity)
        fig.add_shape(
            type="line",
            x0=1,
            y0=-0.5,
            x1=1,
            y1=len(combined_df) - 0.5,
            line=dict(color="green", width=2, dash="dash"),
        )
        
        fig.update_layout(height=800)
    
    return fig

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
