import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare CSV data"""
    print("\nLoading data from CSV...")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    print("\nDataset Overview:")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

class SteelDemandForecaster:
    def __init__(self):
        self.models = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self.feature_sets = {}
        
    def prepare_data(self, df, test_size=0.2):
        """Prepare data for demand forecasting"""
        try:
            print("\nPreparing data for demand forecasting...")
            
            train_size = int(len(df) * (1 - test_size))
            train_df = df[:train_size]
            test_df = df[train_size:]
            
            print(f"\nTrain set size: {len(train_df)}")
            print(f"Test set size: {len(test_df)}")
            
            # Define target variables (only demand)
            target_cols = ['SS316 Demand (MT)', 'SS304 Demand (MT)']
            
            # Define features
            base_features = [
                'Nickel Price (USD/MT)', 
                'Iron Ore Price (USD/MT)', 
                
                'Manufacturing PMI Index',
                'Chromium Price (USD/MT)',
                'Molybdenum Price (USD/MT)',
                'Oil Price (USD/barrel)'
            ]
            
            # Add time-based features
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Year'] = df['Date'].dt.year
            time_features = ['Month', 'Quarter', 'Year']
            
            # Create feature sets for demand forecasting
            self.feature_sets = {
                'SS316 Demand (MT)': base_features + time_features ,
                'SS304 Demand (MT)': base_features + time_features 
            }
            
            X_sets = {}
            y_sets = {}
            
            for target in target_cols:
                current_features = self.feature_sets[target]
                
                X = df[current_features].values
                y = df[target].values.reshape(-1, 1)
                
                # Handle invalid values
                X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
                
                # Create and fit scalers
                self.scalers_X[target] = StandardScaler()
                self.scalers_y[target] = StandardScaler()
                
                X_scaled = self.scalers_X[target].fit_transform(X)
                y_scaled = self.scalers_y[target].fit_transform(y)
                
                X_sets[target] = {
                    'train': X_scaled[:train_size],
                    'test': X_scaled[train_size:],
                    'full': X_scaled
                }
                y_sets[target] = {
                    'train': y_scaled[:train_size],
                    'test': y_scaled[train_size:],
                    'full': y_scaled
                }
                
                print(f"\nFeatures for {target}:")
                print(f"Feature columns: {current_features}")
                print(f"Training shape: {X_sets[target]['train'].shape}")
                print(f"Testing shape: {X_sets[target]['test'].shape}")
            
            return X_sets, y_sets, target_cols
                    
        except Exception as e:
            raise Exception(f"Error preparing data: {str(e)}")

    def train_and_evaluate(self, X_sets, y_sets, target_cols):
        """Train XGBoost models and evaluate performance"""
        print("\nTraining and evaluating XGBoost models...")
        
        results = {}
        for target in target_cols:
            print(f"\nProcessing {target}")
            
            # Initialize model
            model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=7,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Train model
            model.fit(
                X_sets[target]['train'], 
                y_sets[target]['train'].ravel(),
                eval_set=[(X_sets[target]['train'], y_sets[target]['train'].ravel())],
                early_stopping_rounds=50,
                verbose=100
            )
            
            self.models[target] = model
            
            # Evaluate on test set
            y_pred = model.predict(X_sets[target]['test'])
            y_true = self.scalers_y[target].inverse_transform(y_sets[target]['test'])
            y_pred = self.scalers_y[target].inverse_transform(y_pred.reshape(-1, 1))
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Perform time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                model, 
                X_sets[target]['full'], 
                y_sets[target]['full'].ravel(),
                cv=tscv,
                scoring='r2'
            )
            
            results[target] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_scores': cv_scores
            }
            
            # Print detailed results
            print(f"\nEvaluation Metrics for {target}:")
            print(f"MAE: {mae:.2f}")
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R² Score: {r2:.4f}")
            # print("\nCross-validation R² scores:")
            # print(f"Individual folds: {cv_scores}")
            # print(f"Mean CV R²: {cv_scores.mean():.4f}")
            # print(f"Std CV R²: {cv_scores.std():.4f}")
            # 
        return results

    def save_models(self, filename):
        """Save models and scalers"""
        import joblib
        
        joblib.dump(self.models, f"{filename}_xgboost.pkl")
        joblib.dump(self.scalers_X, f"{filename}_scalers_X.pkl")
        joblib.dump(self.scalers_y, f"{filename}_scalers_y.pkl")
        joblib.dump(self.feature_sets, f"{filename}_feature_sets.pkl")
        
        print(f"\nModels and scalers saved with prefix: {filename}")

# Main execution
if __name__ == "__main__":
    print("Steel Demand Forecasting System")
    print("=" * 50)

    try:
        # Load data
        df = load_and_prepare_data('data.csv')
        
        # Initialize forecaster
        print("\nInitializing Steel Demand Forecaster...")
        forecaster = SteelDemandForecaster()

        # Prepare data
        X_sets, y_sets, target_cols = forecaster.prepare_data(df)

        # Train and evaluate models
        results = forecaster.train_and_evaluate(X_sets, y_sets, target_cols)

        # Save models
        forecaster.save_models('steel_demand_models')

        print("\nForecasting system execution completed successfully!")

    except Exception as e:
        print(f"\nError in execution: {str(e)}")
    
    finally:
        print("\nExecution finished.")