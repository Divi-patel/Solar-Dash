import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, distribution plots will be disabled")
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Solar Generation Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #2E7D32;
    }
    h1 {
        color: #1B5E20;
    }
    h2 {
        color: #2E7D32;
    }
    .site-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_site_registry():
    """Load site registry data"""
    registry_path = Path('site_registry.csv')
    if registry_path.exists():
        df = pd.read_csv(registry_path)
        # Clean column names and convert capacity to numeric
        df['dc_capacity_mw'] = pd.to_numeric(df['dc_capacity_mw'], errors='coerce')
        df['ac_capacity_mw'] = pd.to_numeric(df['ac_capacity_mw'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        return df
    return None

@st.cache_data
def find_all_sites():
    """Find all available sites in the Simulation folder"""
    simulation_path = Path('Simulation')
    timeseries_folders = list(simulation_path.glob('*_timeseries'))
    
    sites = []
    for folder in timeseries_folders:
        site_name = folder.name.replace('_timeseries', '')
        sites.append(site_name)
    
    return sorted(sites)

@st.cache_data
def load_site_data(site_name):
    """Load timeseries data for a specific site"""
    folder_path = Path('Simulation') / f'{site_name}_timeseries'
    monthly_files = list(folder_path.glob(f'{site_name}_monthly_timeseries*.csv'))
    
    if monthly_files:
        df = pd.read_csv(monthly_files[0])
        # Get year columns
        year_cols = [col for col in df.columns if str(col).isdigit()]
        return df, year_cols
    return None, []

def create_site_map(registry_df, selected_site=None):
    """Create interactive map of all solar sites"""
    # Filter out any rows with invalid coordinates
    map_df = registry_df.dropna(subset=['latitude', 'longitude'])
    
    # Get sites with data
    sites_with_data = find_all_sites()
    
    # Create hover text
    map_df['hover_text'] = map_df.apply(lambda row: (
        '<b>' + row['site_name'] + '</b><br>' +
        'State: ' + row['state'] + '<br>' +
        'County: ' + row['county'] + '<br>' +
        'DC Capacity: ' + str(round(row['dc_capacity_mw'], 1)) + ' MW<br>' +
        'AC Capacity: ' + str(round(row['ac_capacity_mw'], 1)) + ' MW<br>' +
        'System: ' + row['system_type'].replace('_', ' ').title() + '<br>' +
        'Start: ' + str(row['start_month']) + ' ' + str(row['start_year']) + '<br>' +
        ('<b>Click to analyze</b>' if row['site_name'] in sites_with_data else '<i>No data available</i>')
    ), axis=1)
    
    # Create size based on DC capacity (scaled for visibility)
    map_df['marker_size'] = np.sqrt(map_df['dc_capacity_mw']) * 5 + 10
    
    # Check which sites have data
    map_df['has_data'] = map_df['site_name'].isin(sites_with_data)
    
    # Color by system type
    color_map = {
        'fixed': '#2E7D32',  # Dark green
        'single_axis_tracking': '#4472C4'  # Blue
    }
    map_df['color'] = map_df['system_type'].map(color_map).fillna('#888888')
    
    # Set opacity based on data availability
    map_df['opacity'] = map_df['has_data'].map({True: 0.8, False: 0.3})
    
    # Create the map
    fig = go.Figure()
    
    # Add markers for each system type
    for system_type, color in color_map.items():
        df_filtered = map_df[map_df['system_type'] == system_type]
        if len(df_filtered) > 0:
            # Separate sites with and without data for different opacity
            for has_data in [True, False]:
                df_subset = df_filtered[df_filtered['has_data'] == has_data]
                if len(df_subset) > 0:
                    fig.add_trace(go.Scattermapbox(
                        lat=df_subset['latitude'],
                        lon=df_subset['longitude'],
                        mode='markers',
                        marker=dict(
                            size=df_subset['marker_size'],
                            color=color,
                            opacity=0.8 if has_data else 0.3,
                            sizemode='diameter'
                        ),
                        text=df_subset['hover_text'],
                        hoverinfo='text',
                        name=f"{system_type.replace('_', ' ').title()}{'' if has_data else ' (No Data)'}",
                        customdata=df_subset['site_name'],
                        showlegend=has_data  # Only show in legend if has data
                    ))
    
    # Highlight selected site if any
    if selected_site and selected_site in map_df['site_name'].values:
        selected = map_df[map_df['site_name'] == selected_site].iloc[0]
        fig.add_trace(go.Scattermapbox(
            lat=[selected['latitude']],
            lon=[selected['longitude']],
            mode='markers',
            marker=dict(
                size=selected['marker_size'] + 10,
                color='red',
                opacity=1,
                sizemode='diameter'
            ),
            text=f"<b>SELECTED: {selected['site_name']}</b>",
            hoverinfo='text',
            showlegend=False
        ))
    
    # Set map layout
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(
                lat=map_df['latitude'].mean(),
                lon=map_df['longitude'].mean()
            ),
            zoom=4 if len(map_df) > 1 else 8
        ),
        height=600,
        margin={"r":0,"t":30,"l":0,"b":0},
        title=dict(
            text="Solar Assets Map - Click a site to analyze",
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hovermode='closest'
    )
    
    return fig

def plot_monthly_generation(site_name, df, year_cols):
    """Create monthly generation plot with confidence bands"""
    # Calculate statistics
    df['mean'] = df[year_cols].mean(axis=1)
    df['std'] = df[year_cols].std(axis=1)
    df['p5'] = df[year_cols].quantile(0.05, axis=1)
    df['p95'] = df[year_cols].quantile(0.95, axis=1)
    df['upper_bound'] = df['mean'] + df['std']
    df['lower_bound'] = df['mean'] - df['std']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    x_positions = range(len(df))
    
    # Plot individual years
    for year in year_cols:
        ax.plot(x_positions, df[year], 
               color='lightgray', 
               alpha=0.3, 
               linewidth=1,
               zorder=1)
    
    # Add confidence band
    ax.fill_between(x_positions, 
                   df['lower_bound'], 
                   df['upper_bound'],
                   alpha=0.2, 
                   color='#4472C4',
                   label='Confidence Band',
                   zorder=2)
    
    # Add P5 and P95 lines
    ax.plot(x_positions, df['p5'], 
           color='gray', 
           linewidth=1,
           label='P5',
           zorder=3)
    
    ax.plot(x_positions, df['p95'], 
           color='gray', 
           linewidth=1,
           label='P95',
           zorder=3)
    
    # Plot mean
    ax.plot(x_positions, df['mean'], 
           color='#4472C4',
           linewidth=3,
           label='Mean Forecast',
           zorder=4)
    
    # Styling
    years_range = f"{min(year_cols)}-{max(year_cols)}"
    ax.set_title(f'Solar Generation Forecast - {site_name}\n({years_range})', 
               fontsize=16, 
               fontweight='normal',
               pad=20,
               loc='center')
    
    ax.set_ylabel('MWh', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['month_name'].tolist(), rotation=-45, ha='left')
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    ax.grid(False, axis='x')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    ax.legend(loc='upper center', 
             bbox_to_anchor=(0.5, -0.15),
             ncol=4,
             frameon=False,
             fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def plot_enhanced_distributions(site_name, df, year_cols):
    """Create enhanced distribution plot with gradient shading"""
    if not HAS_SCIPY:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, 'Distribution plot requires scipy package\nPlease install: pip install scipy', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(f'Enhanced Monthly Generation Distributions - {site_name}', fontsize=18)
        return fig
    # Ensure month names exist
    if 'month_name' not in df.columns:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        if 'month' in df.columns:
            df['month_name'] = df['month'].apply(lambda x: month_names[int(x)-1] if 1 <= x <= 12 else f'M{x}')
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Color setup
    colors = ['#E8F5E9', '#A5D6A7', '#66BB6A', '#43A047', '#2E7D32', '#1B5E20']
    
    # Process each month
    for idx, row in df.iterrows():
        # Extract valid data
        month_values = []
        for col in year_cols:
            val = row[col]
            if pd.notna(val) and isinstance(val, (int, float)):
                month_values.append(float(val))
        
        month_data = np.array(month_values)
        if len(month_data) < 2:
            continue
        
        # Calculate statistics
        mean_val = np.mean(month_data)
        std_val = np.std(month_data)
        
        # Calculate percentiles
        p5, p25, p50, p75, p95 = np.percentile(month_data, [5, 25, 50, 75, 95])
        
        # Create KDE
        kde = gaussian_kde(month_data)
        
        # Extend range
        value_min = month_data.min() - std_val * 0.5
        value_max = month_data.max() + std_val * 0.5
        value_range = np.linspace(value_min, value_max, 300)
        
        # Calculate density
        density = kde(value_range)
        
        # Normalize density
        max_width = 0.45
        density_normalized = density / density.max() * max_width
        
        # Create gradient shading
        regions = [
            (value_range <= p5, 0),
            ((value_range > p5) & (value_range <= p25), 1),
            ((value_range > p25) & (value_range <= p50), 2),
            ((value_range > p50) & (value_range <= p75), 3),
            ((value_range > p75) & (value_range <= p95), 4),
            (value_range > p95, 5)
        ]
        
        # Draw each region
        for condition, color_idx in regions:
            mask = condition
            if np.any(mask):
                x_coords = np.concatenate([
                    [idx] * np.sum(mask),
                    idx + density_normalized[mask]
                ])
                y_coords = np.concatenate([
                    value_range[mask],
                    value_range[mask]
                ])
                
                vertices = list(zip(x_coords[:len(x_coords)//2], y_coords[:len(y_coords)//2]))
                vertices.extend(list(zip(x_coords[len(x_coords)//2:][::-1], y_coords[len(y_coords)//2:][::-1])))
                
                ax.fill(*zip(*vertices), color=colors[color_idx], 
                       alpha=0.8, edgecolor='none')
        
        # Add edge line
        ax.plot(idx + density_normalized, value_range, 
               color='#1B5E20', linewidth=1.5, alpha=0.8)
        
        # Add mean line
        mean_idx = np.argmin(np.abs(value_range - mean_val))
        mean_width = density_normalized[mean_idx]
        ax.plot([idx, idx + mean_width], [mean_val, mean_val],
               color='#E74C3C', linewidth=3, zorder=5)
        
        # Add P5 and P95 markers
        p5_idx = np.argmin(np.abs(value_range - p5))
        p95_idx = np.argmin(np.abs(value_range - p95))
        ax.plot([idx, idx + density_normalized[p5_idx]], [p5, p5],
               color='gray', linewidth=1, alpha=0.5)
        ax.plot([idx, idx + density_normalized[p95_idx]], [p95, p95],
               color='gray', linewidth=1, alpha=0.5)
    
    # Styling
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['month_name'].tolist(), fontsize=11, rotation=-45, ha='right')
    
    ax.set_ylabel('Monthly Generation (MWh)', fontsize=13, fontweight='normal')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    ax.set_title(f'Enhanced Monthly Generation Distributions - {site_name}', 
                fontsize=18, fontweight='normal', pad=25, loc='center')
    
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.grid(False, axis='x')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    
    # Create legend
    legend_elements = []
    percentile_labels = ['< P5', 'P5-P25', 'P25-P50', 'P50-P75', 'P75-P95', '> P95']
    for i, (color, label) in enumerate(zip(colors, percentile_labels)):
        legend_elements.append(mpatches.Rectangle((0, 0), 1, 1, 
                                                 facecolor=color, 
                                                 edgecolor='none',
                                                 label=label))
    
    legend_elements.append(plt.Line2D([0], [0], color='#E74C3C', 
                                     linewidth=3, label='Mean'))
    
    legend = ax.legend(handles=legend_elements, 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, -0.1),
                      ncol=7,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      title='Percentile Regions',
                      title_fontsize=12,
                      columnspacing=1.5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def plot_comparison_all_sites():
    """Create comparison plot for all sites"""
    sites = find_all_sites()
    all_sites_data = []
    
    for site_name in sites:
        df, year_cols = load_site_data(site_name)
        if df is not None and year_cols:
            df['mean'] = df[year_cols].mean(axis=1)
            df['site'] = site_name
            all_sites_data.append(df[['month', 'month_name', 'mean', 'site']])
    
    if not all_sites_data:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', 
              '#70AD47', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, site_df in enumerate(all_sites_data):
        site_name = site_df['site'].iloc[0]
        x_positions = range(len(site_df))
        
        ax.plot(x_positions, site_df['mean'], 
               label=site_name, 
               color=colors[i % len(colors)], 
               linewidth=2.5)
    
    ax.set_title('Solar Generation Forecast Comparison\nAll Sites - Mean Monthly Generation', 
                fontsize=16, 
                fontweight='normal',
                pad=20,
                loc='center')
    
    ax.set_ylabel('MWh', fontsize=12)
    
    if all_sites_data:
        x_positions = range(len(all_sites_data[0]))
        month_labels = all_sites_data[0]['month_name'].tolist()
        ax.set_xticks(x_positions)
        ax.set_xticklabels(month_labels, rotation=-45, ha='left')
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    ax.grid(False, axis='x')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    ax.legend(loc='upper center', 
             bbox_to_anchor=(0.5, -0.15),
             ncol=min(5, len(all_sites_data)),
             frameon=False,
             fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def create_plotly_interactive(site_name, df, year_cols):
    """Create interactive Plotly version"""
    # Calculate statistics
    df['mean'] = df[year_cols].mean(axis=1)
    df['std'] = df[year_cols].std(axis=1)
    df['p5'] = df[year_cols].quantile(0.05, axis=1)
    df['p95'] = df[year_cols].quantile(0.95, axis=1)
    df['upper_bound'] = df['mean'] + df['std']
    df['lower_bound'] = df['mean'] - df['std']
    
    month_labels = df['month_name'].tolist()
    
    fig = go.Figure()
    
    # Add individual year traces
    for year in year_cols:
        fig.add_trace(
            go.Scatter(
                x=month_labels,
                y=df[year],
                mode='lines',
                line=dict(color='lightgray', width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Add confidence band
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=df['lower_bound'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(68, 114, 196, 0.2)',
            line=dict(width=0),
            name='Confidence Band',
            hoverinfo='skip'
        )
    )
    
    # Add P5 and P95
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=df['p5'],
            mode='lines',
            name='P5',
            line=dict(color='gray', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=df['p95'],
            mode='lines',
            name='P95',
            line=dict(color='gray', width=1)
        )
    )
    
    # Add mean
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=df['mean'],
            mode='lines',
            name='Mean Forecast',
            line=dict(color='#4472C4', width=3),
            hovertemplate='%{y:,.0f} MWh<extra></extra>'
        )
    )
    
    # Update layout
    years_range = f"{min(year_cols)}-{max(year_cols)}"
    fig.update_layout(
        title=dict(
            text=f"Solar Generation Forecast - {site_name}<br><sub>{years_range}</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="",
        yaxis_title="MWh",
        height=600,
        width=1000,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Main Streamlit App
def main():
    st.title("☀️ Solar Generation Dashboard")
    st.markdown("---")
    
    # Load site registry
    registry_df = load_site_registry()
    
    # Initialize session state
    if 'selected_site' not in st.session_state:
        st.session_state.selected_site = None
    if 'page' not in st.session_state:
        st.session_state.page = 'map'
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🗺️ Navigation")
        
        page_options = ["Site Map", "Site Analysis", "All Sites Comparison"]
        page = st.radio("Select Page", page_options)
        
        if page == "Site Map":
            st.session_state.page = 'map'
        elif page == "All Sites Comparison":
            st.session_state.page = 'comparison'
        else:
            st.session_state.page = 'analysis'
        
        st.markdown("---")
        
        # Site selection for analysis
        if st.session_state.page == 'analysis':
            sites = find_all_sites()
            
            if not sites:
                st.error("No solar sites found in the Simulation folder!")
                return
            
            # Use the selected site from map if available
            if st.session_state.selected_site and st.session_state.selected_site in sites:
                default_index = sites.index(st.session_state.selected_site)
            else:
                default_index = 0
            
            selected_site = st.selectbox(
                "Select Solar Asset",
                options=sites,
                index=default_index,
                help="Choose a site to analyze"
            )
            
            # Update session state
            st.session_state.selected_site = selected_site
            
            # Plot type selection
            plot_options = [
                "Monthly Generation Forecast",
                "Interactive Plot (Plotly)"
            ]
            if HAS_SCIPY:
                plot_options.insert(1, "Enhanced Distribution Plot")
            
            plot_type = st.selectbox(
                "Select Visualization Type",
                options=plot_options,
                help="Choose the type of visualization"
            )
        
        st.markdown("---")
        
        # Information section
        st.info(
            "**Navigation Guide:**\n\n"
            "🗺️ **Site Map**: Interactive map of all solar assets\n\n"
            "📊 **Site Analysis**: Detailed analysis for individual sites\n\n"
            "📈 **All Sites Comparison**: Compare generation across all sites"
        )
    
    # Main content area
    if st.session_state.page == 'map':
        st.header("🗺️ Solar Assets Map")
        
        if registry_df is not None and not registry_df.empty:
            # Show map stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sites", len(registry_df))
            with col2:
                total_dc = registry_df['dc_capacity_mw'].sum()
                st.metric("Total DC Capacity", f"{total_dc:,.0f} MW")
            with col3:
                total_ac = registry_df['ac_capacity_mw'].sum()
                st.metric("Total AC Capacity", f"{total_ac:,.0f} MW")
            with col4:
                states = registry_df['state'].nunique()
                st.metric("States Covered", states)
            
            st.markdown("---")
            
            # Add instructions
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("💡 **Tip:** Click on any marker on the map to analyze that site, or use the dropdown below. Faded markers indicate sites without data yet.")
            
            # Create and display map
            map_fig = create_site_map(registry_df, st.session_state.selected_site)
            
            # Handle map clicks
            clicked_point = st.plotly_chart(map_fig, use_container_width=True, on_select="rerun", key="site_map")
            
            # Check if a point was clicked
            if clicked_point and 'selection' in clicked_point and clicked_point['selection']['points']:
                try:
                    # Get the clicked point data
                    point_data = clicked_point['selection']['points'][0]
                    # The customdata contains the site name
                    if 'customdata' in point_data:
                        clicked_site = point_data['customdata']
                        if isinstance(clicked_site, list):
                            clicked_site = clicked_site[0]
                        st.session_state.selected_site = clicked_site
                        st.session_state.page = 'analysis'
                        st.rerun()
                except Exception as e:
                    st.error(f"Error selecting site: {str(e)}")
            
            # Alternative site selection dropdown
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                sites_in_registry = registry_df['site_name'].tolist()
                sites_with_data = find_all_sites()
                available_sites = [site for site in sites_with_data if site in sites_in_registry or True]  # Show all sites with data
                
                selected_from_dropdown = st.selectbox(
                    "Or select a site:",
                    options=[''] + available_sites,
                    key="map_site_selector"
                )
                
                if selected_from_dropdown and selected_from_dropdown != '':
                    st.session_state.selected_site = selected_from_dropdown
                    st.session_state.page = 'analysis'
                    st.rerun()
            
            # Show site cards below map
            st.markdown("---")
            st.subheader("📋 Site Directory")
            
            # Filter by state
            states = ['All'] + sorted(registry_df['state'].unique().tolist())
            selected_state = st.selectbox("Filter by State", states)
            
            # Filter dataframe
            if selected_state != 'All':
                filtered_df = registry_df[registry_df['state'] == selected_state]
            else:
                filtered_df = registry_df
            
            # Display site cards in columns
            sites_with_data = find_all_sites()
            cols = st.columns(2)
            for idx, (_, site) in enumerate(filtered_df.iterrows()):
                with cols[idx % 2]:
                    with st.container():
                        # Check if this site has data
                        has_data = site['site_name'] in sites_with_data
                        
                        st.markdown(f"""
                        <div class="site-card">
                            <h4>{site['site_name']}</h4>
                            <p><strong>Location:</strong> {site['county']}, {site['state']}<br>
                            <strong>Capacity:</strong> {site['dc_capacity_mw']:.1f} MW DC / {site['ac_capacity_mw']:.1f} MW AC<br>
                            <strong>System:</strong> {site['system_type'].replace('_', ' ').title()}<br>
                            <strong>Started:</strong> {site['start_month']} {site['start_year']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if has_data:
                            if st.button(f"Analyze {site['site_name']}", key=f"btn_{site['plant_code']}"):
                                st.session_state.selected_site = site['site_name']
                                st.session_state.page = 'analysis'
                                st.rerun()
                        else:
                            st.warning("No data available for this site")
        else:
            st.error("No site registry data found. Please ensure 'site_registry.csv' exists in the root directory.")
    
    elif st.session_state.page == 'comparison':
        st.header("📊 All Sites Comparison")
        with st.spinner("Creating comparison plot..."):
            fig = plot_comparison_all_sites()
            if fig:
                st.pyplot(fig)
                
                # Show comparison statistics
                st.markdown("---")
                st.subheader("📈 Comparison Statistics")
                
                sites = find_all_sites()
                comparison_data = []
                
                for site_name in sites:
                    df, year_cols = load_site_data(site_name)
                    if df is not None and year_cols:
                        annual_mean = df[year_cols].sum().mean()
                        annual_std = df[year_cols].sum().std()
                        comparison_data.append({
                            'Site': site_name,
                            'Mean Annual Generation (MWh)': annual_mean,
                            'Std Dev (MWh)': annual_std,
                            'CV%': (annual_std / annual_mean * 100) if annual_mean > 0 else 0
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    comp_df = comp_df.sort_values('Mean Annual Generation (MWh)', ascending=False)
                    
                    # Format numbers
                    comp_df['Mean Annual Generation (MWh)'] = comp_df['Mean Annual Generation (MWh)'].round(0)
                    comp_df['Std Dev (MWh)'] = comp_df['Std Dev (MWh)'].round(0)
                    comp_df['CV%'] = comp_df['CV%'].round(1)
                    
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.error("No data available for comparison")
    
    else:  # analysis page
        if st.session_state.selected_site:
            # Load data for selected site
            df, year_cols = load_site_data(st.session_state.selected_site)
            
            if df is None or not year_cols:
                st.error(f"No data found for {st.session_state.selected_site}")
                return
            
            # Display site info with registry data if available
            st.header(f"📍 {st.session_state.selected_site}")
            
            # Show site details from registry if available
            if registry_df is not None and st.session_state.selected_site in registry_df['site_name'].values:
                site_info = registry_df[registry_df['site_name'] == st.session_state.selected_site].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    **Location:** {site_info['county']}, {site_info['state']}  
                    **Coordinates:** {site_info['latitude']:.4f}, {site_info['longitude']:.4f}
                    """)
                with col2:
                    st.markdown(f"""
                    **DC Capacity:** {site_info['dc_capacity_mw']:.1f} MW  
                    **AC Capacity:** {site_info['ac_capacity_mw']:.1f} MW
                    """)
                with col3:
                    st.markdown(f"""
                    **System Type:** {site_info['system_type'].replace('_', ' ').title()}  
                    **Started:** {site_info['start_month']} {site_info['start_year']}
                    """)
                
                st.markdown("---")
            
            # Show basic statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            mean_annual = df[year_cols].sum().mean()
            max_month = df[year_cols].max().max()
            min_month = df[year_cols].min().min()
            years_available = len(year_cols)
            
            with col_stat1:
                st.metric("Years of Data", years_available)
            with col_stat2:
                st.metric("Avg Annual Generation", f"{mean_annual:,.0f} MWh")
            with col_stat3:
                st.metric("Max Monthly", f"{max_month:,.0f} MWh")
            with col_stat4:
                st.metric("Min Monthly", f"{min_month:,.0f} MWh")
            
            st.markdown("---")
            
            # Create selected visualization
            if plot_type == "Monthly Generation Forecast":
                with st.spinner("Creating forecast plot..."):
                    fig = plot_monthly_generation(st.session_state.selected_site, df, year_cols)
                    st.pyplot(fig)
                    
            elif plot_type == "Enhanced Distribution Plot":
                with st.spinner("Creating distribution plot..."):
                    fig = plot_enhanced_distributions(st.session_state.selected_site, df, year_cols)
                    st.pyplot(fig)
                    
            elif plot_type == "Interactive Plot (Plotly)":
                with st.spinner("Creating interactive plot..."):
                    fig = create_plotly_interactive(st.session_state.selected_site, df, year_cols)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Add download option for data
            st.markdown("---")
            with st.expander("📥 Download Data"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{st.session_state.selected_site}_monthly_data.csv",
                    mime="text/csv"
                )
            
            # Add data table view
            st.markdown("---")
            with st.expander("📊 View Data Table", expanded=True):
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Full Data", "Statistical Summary", "Yearly Comparison"])
                
                with tab1:
                    st.subheader("Full Monthly Time Series Data")
                    # Show the dataframe with month names as index for better readability
                    display_df = df.set_index('month_name')
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400
                    )
                
                with tab2:
                    st.subheader("Statistical Summary")
                    # Calculate statistics for each month
                    stats_df = pd.DataFrame({
                        'Month': df['month_name'],
                        'Mean': df[year_cols].mean(axis=1).round(1),
                        'Std Dev': df[year_cols].std(axis=1).round(1),
                        'Min': df[year_cols].min(axis=1).round(1),
                        'P25': df[year_cols].quantile(0.25, axis=1).round(1),
                        'Median': df[year_cols].quantile(0.50, axis=1).round(1),
                        'P75': df[year_cols].quantile(0.75, axis=1).round(1),
                        'Max': df[year_cols].max(axis=1).round(1),
                        'CV%': (df[year_cols].std(axis=1) / df[year_cols].mean(axis=1) * 100).round(1)
                    })
                    
                    st.dataframe(
                        stats_df.set_index('Month'),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Add explanation
                    st.caption("**CV%** = Coefficient of Variation (Std Dev / Mean × 100) - shows relative variability")
                
                with tab3:
                    st.subheader("Year-over-Year Comparison")
                    # Transpose for yearly view
                    yearly_df = df[['month_name'] + year_cols].set_index('month_name').T
                    yearly_df.index.name = 'Year'
                    
                    # Calculate yearly totals
                    yearly_df['Annual Total'] = yearly_df.sum(axis=1).round(0)
                    
                    # Show recent years first
                    st.dataframe(
                        yearly_df.iloc[::-1],  # Reverse to show recent years first
                        use_container_width=True,
                        height=400
                    )
                    
                    # Add yearly statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        best_year = yearly_df['Annual Total'].idxmax()
                        best_value = yearly_df['Annual Total'].max()
                        st.metric("Best Year", f"{best_year}", f"{best_value:,.0f} MWh")
                    
                    with col2:
                        worst_year = yearly_df['Annual Total'].idxmin()
                        worst_value = yearly_df['Annual Total'].min()
                        st.metric("Worst Year", f"{worst_year}", f"{worst_value:,.0f} MWh")
                    
                    with col3:
                        yearly_std = yearly_df['Annual Total'].std()
                        st.metric("Yearly Std Dev", f"{yearly_std:,.0f} MWh")
        else:
            st.info("Please select a site from the map or sidebar to begin analysis.")

if __name__ == "__main__":
    main()
