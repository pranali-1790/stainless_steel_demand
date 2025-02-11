import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_steel_data():
    # Generate dates from 2020 to Jan 2025
    dates = pd.date_range(start='2020-01-01', end='2025-01-31', freq='M')
    df = pd.DataFrame(index=dates)
    
    # Raw Material Prices
    # Nickel Price (USD/MT) - Critical for both grades
    nickel_base = 15000 + np.random.normal(0, 500, len(dates))
    covid_impact = -2000 * np.exp(-((dates.year - 2020) ** 2))
    nickel_trend = 2000 * np.sin(np.pi * (dates.year + dates.month/12 - 2020)/2)
    df['Nickel Price (USD/MT)'] = nickel_base + covid_impact + nickel_trend
    df.loc['2022-03':'2022-05', 'Nickel Price (USD/MT)'] += 30000  # Historical spike
    
    # Override 2024 and Jan 2025 nickel prices with provided values
    nickel_2024_2025 = [16500, 16200, 15900, 15600, 15400, 15200, 15100, 15000, 14900, 14800, 14700, 14600, 15482]
    df.loc['2024-01':'2025-01', 'Nickel Price (USD/MT)'] = nickel_2024_2025
    
    # Chromium Price (USD/MT) - Essential for both grades
    df['Chromium Price (USD/MT)'] = 11000 + 1000 * np.sin(np.pi * (dates.year + dates.month/12 - 2020)/2) + \
                          np.random.normal(0, 200, len(dates))
    
    # Molybdenum Price (USD/MT) - Critical for SS316
    df['Molybdenum Price (USD/MT)'] = (40 + 5 * np.sin(np.pi * (dates.year + dates.month/12 - 2020)/2) + \
                            np.random.normal(0, 1, len(dates))) * 1000
    
    # Global Economic Indicators
    # Manufacturing PMI Index
    pmi_base = 52 + np.random.normal(0, 1, len(dates))
    covid_impact = -15 * np.exp(-((dates.year - 2020) ** 2))
    recovery = 5 * (1 - np.exp(-(dates.year - 2020)))
    df['Manufacturing PMI Index'] = pmi_base + covid_impact + recovery
    df['Manufacturing PMI Index'] = df['Manufacturing PMI Index'].clip(35, 65)
    
    # Global Oil Price (USD/barrel)
    oil_base = 70 + np.random.normal(0, 5, len(dates))
    oil_trend = 20 * np.sin(np.pi * (dates.year + dates.month/12 - 2020)/2)
    df['Oil Price (USD/barrel)'] = oil_base + oil_trend
    df.loc['2022-02':'2022-06', 'Oil Price (USD/barrel)'] += 30
    
    # Iron Ore Price (USD/MT)
    iron_base = 100 + np.random.normal(0, 10, len(dates))
    iron_trend = 50 * np.sin(np.pi * (dates.year + dates.month/12 - 2020)/2)
    df['Iron Ore Price (USD/MT)'] = iron_base + iron_trend
    df.loc['2021-05':'2021-08', 'Iron Ore Price (USD/MT)'] += 120
    
    # SS316 Price (USD/MT)
    df['SS316 Price (USD/MT)'] = 3000 + \
                       0.15 * df['Nickel Price (USD/MT)'] + \
                       0.1 * df['Chromium Price (USD/MT)'] + \
                       0.05 * df['Molybdenum Price (USD/MT)'] + \
                       5 * df['Oil Price (USD/barrel)'] + \
                       np.random.normal(0, 100, len(dates))
    
    # SS304 Price (USD/MT)
    df['SS304 Price (USD/MT)'] = 2400 + \
                       0.12 * df['Nickel Price (USD/MT)'] + \
                       0.08 * df['Chromium Price (USD/MT)'] + \
                       4 * df['Oil Price (USD/barrel)'] + \
                       np.random.normal(0, 80, len(dates))
    
    # Modified SS316 Demand with stronger inverse price relationship
    base_demand_316 = 50000 + np.random.normal(0, 1000, len(dates))
    seasonal_316 = 5000 * np.sin(2 * np.pi * dates.month/12)
    # Strengthened inverse price elasticity
    price_elasticity_316 = -0.8 * (df['Nickel Price (USD/MT)'] - df['Nickel Price (USD/MT)'].mean()) / df['Nickel Price (USD/MT)'].std()
    pmi_effect_316 = 2000 * (df['Manufacturing PMI Index'] - 50) / 10
    df['SS316 Demand (MT)'] = base_demand_316 + seasonal_316 + price_elasticity_316 * 2000 + pmi_effect_316
    
    # Modified SS304 Demand with stronger inverse price relationship
    base_demand_304 = 120000 + np.random.normal(0, 2000, len(dates))
    seasonal_304 = 12000 * np.sin(2 * np.pi * dates.month/12)
    # Strengthened inverse price elasticity
    price_elasticity_304 = -0.9 * (df['Nickel Price (USD/MT)'] - df['Nickel Price (USD/MT)'].mean()) / df['Nickel Price (USD/MT)'].std()
    pmi_effect_304 = 5000 * (df['Manufacturing PMI Index'] - 50) / 10
    df['SS304 Demand (MT)'] = base_demand_304 + seasonal_304 + price_elasticity_304 * 4000 + pmi_effect_304
    
    # Reorder columns
    column_order = [
        'Nickel Price (USD/MT)',
        'Chromium Price (USD/MT)',
        'Molybdenum Price (USD/MT)',
        'Iron Ore Price (USD/MT)',
        'Oil Price (USD/barrel)',
        'Manufacturing PMI Index',
        'SS316 Price (USD/MT)',
        'SS304 Price (USD/MT)',
        'SS316 Demand (MT)',
        'SS304 Demand (MT)'
    ]
    
    df = df[column_order]
    
    # Round all values
    df = df.round({
        'Nickel Price (USD/MT)': 0,
        'Chromium Price (USD/MT)': 0,
        'Molybdenum Price (USD/MT)': 0,
        'Iron Ore Price (USD/MT)': 1,
        'Oil Price (USD/barrel)': 1,
        'Manufacturing PMI Index': 1,
        'SS316 Price (USD/MT)': 0,
        'SS304 Price (USD/MT)': 0,
        'SS316 Demand (MT)': 0,
        'SS304 Demand (MT)': 0
    })
    
    return df

# Generate the data
steel_data = generate_steel_data()

# Export to CSV
steel_data.to_csv('data3.csv', date_format='%Y-%m-%d')

# Display 2024-2025 data
print(steel_data['2024':'2025'])