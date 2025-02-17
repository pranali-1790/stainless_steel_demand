import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
from dateutil.relativedelta import relativedelta



# Page configuration
st.set_page_config(
    page_title="Steel Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        color: #0066cc;
    }
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-unit {
        font-size: 0.8em;
        color: #666;
    }
    .metric-change {
        font-size: 1em;
        margin-top: 5px;
    }
    .positive-change {
        color: #2ecc71;
    }
    .negative-change {
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)


def create_month_year_options(start_date, end_date):
    """Create a list of month-year options between June 2024 and December 2025"""
    dates = pd.date_range(start='2024-06-01', end='2025-12-31', freq='ME')
    return [(d.strftime('%b %Y'), d) for d in dates] 

@st.cache_data
def load_data():
    """Load historical data and trained models"""
    # Load historical data
    df = pd.read_csv('data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load models and scalers
    models = joblib.load('steel_demand_models_xgboost.pkl')
    scalers_X = joblib.load('steel_demand_models_scalers_X.pkl')
    scalers_y = joblib.load('steel_demand_models_scalers_y.pkl')
    feature_sets = joblib.load('steel_demand_models_feature_sets.pkl')
    
    return df, models, scalers_X, scalers_y, feature_sets

@st.cache_data
def load_external_factors_forecast():
    """Load forecasted values for external factors"""
    forecast_factors = pd.read_csv('external_factors_forecasted_data.csv')
    forecast_factors['Date'] = pd.to_datetime(forecast_factors['Date'])
    return forecast_factors

def prepare_forecast_features(df, start_date, end_date, grade, forecast_factors):
    """Prepare features for forecasting using forecasted external factors"""
    # Create date range for prediction
    date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    forecast_df = pd.DataFrame({'Date': date_range})
    
    # Add time features
    forecast_df['Month'] = forecast_df['Date'].dt.month
    forecast_df['Quarter'] = forecast_df['Date'].dt.quarter
    forecast_df['Year'] = forecast_df['Date'].dt.year
    
    # Define external factor columns
    factor_columns = [
        'Nickel Price (USD/MT)',
        'Iron Ore Price (USD/MT)',
        'Chromium Price (USD/MT)',
        'Molybdenum Price (USD/MT)',
        'Oil Price (USD/barrel)',
        'Manufacturing PMI Index',
        # f'{grade} Price (USD/MT)'
    ]
    
    # For each date in the forecast period
    for date in date_range:
        # If the date is in the forecast_factors DataFrame, use forecasted values
        if date in forecast_factors['Date'].values:
            factor_values = forecast_factors[forecast_factors['Date'] == date][factor_columns].iloc[0]
        else:
            # If no forecast available, use the latest known values from historical data
            factor_values = df[df['Date'] <= date].iloc[-1][factor_columns]
        
        # Update the forecast_df with the appropriate values
        forecast_df.loc[forecast_df['Date'] == date, factor_columns] = factor_values
    
    return forecast_df

def generate_forecast(df, forecast_df, grade, models, scalers_X, scalers_y, feature_sets):
    """Generate demand forecast for specified period"""
    target_col = f'{grade} Demand (MT)'
    features = feature_sets[target_col]
    
    # Create fresh copy to avoid modifications to original
    forecast_df = forecast_df.copy()
    
    # Ensure all required features are present
    missing_features = [f for f in features if f not in forecast_df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Prepare features for prediction
    X = forecast_df[features].values
    
    # If we have only one row, reshape appropriately
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # Handle any NaN values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Transform features
    X_scaled = scalers_X[target_col].transform(X)
    
    # Generate predictions
    y_pred_scaled = models[target_col].predict(X_scaled)
    
    # Reshape predictions for inverse transform if needed
    if len(y_pred_scaled.shape) == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    
    # Inverse transform predictions
    y_pred = scalers_y[target_col].inverse_transform(y_pred_scaled)
    
    # Ensure y_pred is flattened
    y_pred = y_pred.flatten()
    
    # Verify shapes match
    if len(y_pred) != len(forecast_df):
        raise ValueError(f"Shape mismatch: predictions ({len(y_pred)}) != forecast_df ({len(forecast_df)})")
    
    # Add predictions to forecast DataFrame
    forecast_df['Forecasted_Demand'] = y_pred
    
    return forecast_df
        
    


def plot_forecast_vs_actual(historical_df, forecast_df, grade):
    """Create plot comparing forecasted vs actual demand with seamless transition and markers for both lines"""
    fig = go.Figure()
    
    historical_col = f'{grade} Demand (MT)'
    current_date = pd.Timestamp('2025-02-10')
    
    # Get the last point of historical data before forecast starts
    last_historical_point = historical_df[
        historical_df['Date'] <= current_date
    ].iloc[-1]
    
    # Ensure forecast starts from the last historical point
    forecast_df_aligned = forecast_df.copy()
    
    # Add the last historical point to the forecast data
    forecast_df_aligned = pd.concat([
        pd.DataFrame({
            'Date': [last_historical_point['Date']],
            'Forecasted_Demand': [last_historical_point[historical_col]]
        }),
        forecast_df_aligned
    ]).drop_duplicates(subset=['Date'])
    
    # Sort by date to ensure proper line connection
    forecast_df_aligned = forecast_df_aligned.sort_values('Date')
    
    # Plot historical data with markers
    fig.add_trace(go.Scatter(
        x=historical_df[historical_df['Date'] <= current_date]['Date'],
        y=historical_df[historical_df['Date'] <= current_date][historical_col],
        name='Historical Demand',
        line=dict(color='blue', width=2),
        mode='lines+markers',  # Add markers
        marker=dict(
            color='blue',
            size=8,
            symbol='circle'
        )
    ))
    
    # Plot forecast data starting from the last historical point with markers
    fig.add_trace(go.Scatter(
        x=forecast_df_aligned['Date'],
        y=forecast_df_aligned['Forecasted_Demand'],
        name='Forecasted Demand',
        line=dict(color='red', dash='dash', width=2),
        mode='lines+markers',  # Add markers
        marker=dict(
            color='red',
            size=8,
            symbol='circle'
        )
    ))
    
    fig.update_layout(
        title=f'{grade} Demand Forecast',
        xaxis_title='Date',
        yaxis_title='Demand (MT)',
        hovermode='x unified',
        showlegend=True,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            title_text='Demand (MT)'
        ),
        plot_bgcolor='white'
    )
    
    return fig

def create_factor_impact_cards(df, start_date, end_date, selected_grade, forecast_factors):
    """Create cards showing external factor trends with dynamic current values"""
    factors = {
        'Nickel Price': {
            'col': 'Nickel Price (USD/MT)',
            'icon': '🔧',
            'unit': 'USD/MT',
            'description': 'Key Raw Material'
        },
        'Iron Ore Price': {
            'col': 'Iron Ore Price (USD/MT)',
            'icon': '⛰️',
            'unit': 'USD/MT',
            'description': 'Base Raw Material'
        },
        'Chromium Price': {
            'col': 'Chromium Price (USD/MT)',
            'icon': '⚒️',
            'unit': 'USD/MT',
            'description': 'Alloying Element'
        },
        'Molybdenum Price': {
            'col': 'Molybdenum Price (USD/MT)',
            'icon': '🏗️',
            'unit': 'USD/MT',
            'description': 'Alloying Element'
        },
        'Oil Price': {
            'col': 'Oil Price (USD/barrel)',
            'icon': '🛢️',
            'unit': 'USD/barrel',
            'description': 'Energy Cost Factor'
        },
        'Manufacturing PMI': {
            'col': 'Manufacturing PMI Index',
            'icon': '📊',
            'unit': 'Index Value',
            'description': 'Industry Health Indicator'
        }
    }
    
    # Convert dates to timestamps if they aren't already
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Create combined dataset for the selected period
    period_data = pd.DataFrame()
    
    # For each date in the range
    current_date = pd.Timestamp('2025-02-10')  # Current date
    
    if start_date <= current_date:
        # Get historical data up to current date
        historical_data = df[
            (df['Date'] >= start_date) & 
            (df['Date'] <= min(current_date, end_date))
        ]
        period_data = pd.concat([period_data, historical_data])
    
    if end_date > current_date:
        # Get forecast data after current date
        forecast_period = forecast_factors[
            (forecast_factors['Date'] > current_date) & 
            (forecast_factors['Date'] <= end_date)
        ]
        period_data = pd.concat([period_data, forecast_period])
    
    # Sort the combined data by date
    period_data = period_data.sort_values('Date')
    
    # Create two rows of cards, 3 cards per row
    for i in range(0, len(factors), 3):
        cols = st.columns(3)
        for j, (factor_name, factor_info) in enumerate(list(factors.items())[i:i+3]):
            col = factor_info['col']
            
            if not period_data.empty:
                # Get the latest value based on the current date
                if end_date > current_date:
                    # Use forecasted value for future dates
                    current_value = forecast_factors[
                        forecast_factors['Date'] == min(end_date, forecast_factors['Date'].max())
                    ][col].iloc[0]
                else:
                    # Use historical value
                    current_value = period_data[col].iloc[-1]
                
                # Calculate change over the selected period
                start_value = period_data[col].iloc[0]
                change = ((current_value - start_value) / start_value) * 100
                min_value = period_data[col].min()
                max_value = period_data[col].max()
            else:
                # Fallback to latest historical value if no data available
                current_value = df[col].iloc[-1]
                change = 0
                min_value = max_value = current_value
            
            change_class = "positive-change" if change >= 0 else "negative-change"
            
            with cols[j]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{factor_info['icon']} {factor_name}</h3>
                    <p style="color: #666;">{factor_info['description']}</p>
                    <div class="metric-value">
                        {current_value:,.2f} <span class="metric-unit">{factor_info['unit']}</span>
                    </div>
                    <div class="metric-change {change_class}">
                        {change:+.2f}% Change
                    </div>
                    <p style="font-size: 0.8em; color: #666;">
                        Range: {min_value:,.2f} - {max_value:,.2f} {factor_info['unit']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
def main():
    st.markdown('<h1 class="main-header">Steel Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    try:
       
        # Load data and models
        df, models, scalers_X, scalers_y, feature_sets = load_data()

        # Load external factors forecast
        forecast_factors = load_external_factors_forecast()
        
        # Sidebar controls
        st.sidebar.header("Forecast Parameters")
        
        # Steel grade selection
        grade = st.sidebar.selectbox(
            "Select Steel Grade",
            ["SS316", "SS304"]
        )
        
        # Calculate date ranges
        min_date = pd.Timestamp('2024-06-01')
        max_date = pd.Timestamp('2025-12-31')
        
        # Create month-year options with abbreviated months
        month_year_options = create_month_year_options(min_date, max_date)
        month_year_labels = [opt[0] for opt in month_year_options]
        month_year_dates = [opt[1] for opt in month_year_options]
        
        # Find indices for February 2025 and May 2025
        feb_2025_idx = next((i for i, date in enumerate(month_year_dates) if date.strftime('%Y-%m') == '2025-02'), 0)
        may_2025_idx = next((i for i, date in enumerate(month_year_dates) if date.strftime('%Y-%m') == '2025-05'), len(month_year_options)-1)
        
        # Create two columns for start and end month selection
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.markdown("### Start Month")
            start_idx = st.selectbox(
                "Select Start Month",
                range(len(month_year_options)),
                format_func=lambda x: month_year_labels[x],
                key="start_month",
                index=feb_2025_idx,  # Set default to February 2025
                label_visibility="collapsed"
            )

        with col2:
            st.markdown("### End Month")
            end_idx = st.selectbox(
                "Select End Month",
                range(start_idx, len(month_year_options)),
                format_func=lambda x: month_year_labels[x],
                index=may_2025_idx - feb_2025_idx,  # Adjust index relative to start_idx
                key="end_month",
                label_visibility="collapsed"
            )
        # Get selected dates
        start_date = month_year_dates[start_idx]
        end_date = month_year_dates[end_idx]
        
        forecast_df = prepare_forecast_features(df, start_date, end_date, grade, forecast_factors)
        forecast_df = generate_forecast(df, forecast_df, grade, models, scalers_X, scalers_y, feature_sets)
        
        # Display forecast plot
        st.plotly_chart(plot_forecast_vs_actual(df, forecast_df, grade), use_container_width=True)
        
        # Display forecast table with external factors
        st.subheader("Forecast Details")
        
        external_factors = [
            'Nickel Price (USD/MT)',
            'Iron Ore Price (USD/MT)',
            'Chromium Price (USD/MT)',
            'Molybdenum Price (USD/MT)',
            'Oil Price (USD/barrel)',
            'Manufacturing PMI Index',
        ]
        
        # Prepare table data
        table_data = forecast_df[['Date', 'Forecasted_Demand']].copy()

        # Add external factors from forecast_factors for future dates
        for factor in external_factors:
            table_data[factor] = pd.NA  # Initialize with NA
            
            # For each date in table_data
            for idx, row in table_data.iterrows():
                date = row['Date']
                
                if date in forecast_factors['Date'].values:
                    # Use forecasted values for future dates
                    table_data.loc[idx, factor] = forecast_factors[
                        forecast_factors['Date'] == date
                    ][factor].iloc[0]
                elif date in df['Date'].values:
                    # Use historical values for past dates
                    table_data.loc[idx, factor] = df[
                        df['Date'] == date
                    ][factor].iloc[0]
                else:
                    # Use the latest available value as fallback
                    table_data.loc[idx, factor] = df[factor].iloc[-1]

        # Add actual demand if available
        historical_data = df[df['Date'] >= start_date].copy()
        if not historical_data.empty:
            # Convert dates to abbreviated month format
            historical_data['Month'] = historical_data['Date'].dt.strftime('%b %Y')
            table_data['Month'] = table_data['Date'].dt.strftime('%b %Y')
            
            # Add actual demand
            actual_demand_dict = historical_data.set_index('Month')[f'{grade} Demand (MT)'].to_dict()
            table_data['Actual Demand (MT)'] = table_data['Month'].map(actual_demand_dict)
        else:
            table_data['Month'] = table_data['Date'].dt.strftime('%b %Y')
            table_data['Actual Demand (MT)'] = pd.NA

        # Drop the Date column and keep only Month
        table_data = table_data.drop('Date', axis=1)
        table_data = table_data.rename(columns={'Forecasted_Demand': 'Forecasted Demand (MT)'})

        # Remove rows where Forecasted Demand is null
        table_data = table_data.dropna(subset=['Forecasted Demand (MT)'])

        # Reorder columns
        ordered_columns = ['Month', 'Forecasted Demand (MT)', 'Actual Demand (MT)'] + external_factors
        table_data = table_data[ordered_columns]

        # Reset index and drop it
        table_data = table_data.reset_index(drop=True)

        # Create format dictionary for numeric columns
        format_dict = {
            'Forecasted Demand (MT)': '{:.2f}',
            'Actual Demand (MT)': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '-'  # Changed empty string to '-'
        }

        # Add formatting for external factors
        for factor in external_factors:
            format_dict[factor] = '{:.2f}'

        # Display the table with horizontal scrolling
        st.dataframe(
            table_data.style.format(format_dict),
            use_container_width=True,
            height=400
        )
    
    
        # External Factors Impact Section
        st.subheader("External Factors Impact")
        create_factor_impact_cards(df, start_date, end_date, grade,forecast_factors)
        
        # Modified correlation heatmap section
        st.subheader("Factor Correlation Analysis")

        # Define external factors
        external_factors = [
            'Nickel Price (USD/MT)', 
            'Iron Ore Price (USD/MT)',
            'Chromium Price (USD/MT)',
            'Molybdenum Price (USD/MT)',
            'Oil Price (USD/barrel)',
            'Manufacturing PMI Index'
        ]

        # Calculate correlations only between demand and external factors
        correlations = []
        for factor in external_factors:
            corr = df[factor].corr(df[f'{grade} Demand (MT)'])
            correlations.append(corr)

        # Create correlation plot
        fig = go.Figure(data=go.Bar(
            x=external_factors,
            y=correlations,
            text=np.round(correlations, 2),
            textposition='auto',
        ))

        fig.update_layout(
            title=f"Correlation between {grade} Demand and External Factors",
            xaxis_title="External Factors",
            yaxis_title="Correlation Coefficient",
            xaxis_tickangle=-45,
            yaxis=dict(range=[-1, 1]),  # Set y-axis range from -1 to 1
            height=500,
            showlegend=False
        )

        # Add a horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

                # Update the bar chart layout
        fig.update_traces(
            width=0.4,  # Reduce bar width
            marker_color=[
                'red' if x < 0 else 'green' for x in correlations
            ]
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"\nDetailed error information:")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()      