"""
Natural Gas Storage Contract Pricing Model
JPMorgan Chase Quantitative Research Task

This module analyzes monthly natural gas prices, identifies seasonal patterns,
and provides price estimates for any date (past, present, or future).
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


class NaturalGasPricingModel:
    """
    A pricing model for natural gas that handles:
    - Seasonal pattern analysis
    - Historical price interpolation
    - Future price extrapolation
    """
    
    def __init__(self, csv_path):
        """
        Initialize the model with historical data.
        
        Args:
            csv_path (str): Path to the CSV file with monthly prices
        """
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
        self._analyze_seasonality()
        self._build_extrapolation_model()
    
    def _preprocess_data(self):
        """Parse dates and prices, convert to appropriate formats."""
        # Parse dates - handle various date formats
        self.df['Date'] = pd.to_datetime(self.df['Dates'], format='%m/%d/%y')
        self.df['Price'] = pd.to_numeric(self.df['Prices'], errors='coerce')
        
        # Sort by date
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Extract year, month, day of year for analysis
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
        
        # Store data range info
        self.min_date = self.df['Date'].min()
        self.max_date = self.df['Date'].max()
        self.extrapolation_end = self.max_date + timedelta(days=365)
        
        print(f"Data loaded: {self.min_date.date()} to {self.max_date.date()}")
        print(f"Price range: ${self.df['Price'].min():.2f} - ${self.df['Price'].max():.2f}")
        print(f"Number of data points: {len(self.df)}")
    
    def _analyze_seasonality(self):
        """Analyze seasonal patterns in the data."""
        # Calculate average price by month
        self.seasonal_avg = self.df.groupby('Month')['Price'].mean()
        
        # Calculate trend and seasonal components
        # Set frequency for seasonal decomposition (monthly)
        ts_data = self.df.set_index('Date')['Price']
        
        # Use additive decomposition with appropriate period
        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=12, extrapolate='extend')
        except TypeError:
            # Older version of statsmodels
            decomposition = seasonal_decompose(ts_data, model='additive', period=12)
        
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid
        
        print("\n--- Seasonal Analysis ---")
        print("Average Price by Month:")
        for month in range(1, 13):
            month_name = pd.Timestamp(2024, month, 1).strftime('%B')
            avg_price = self.seasonal_avg.get(month, np.nan)
            if not np.isnan(avg_price):
                print(f"  {month_name:12s}: ${avg_price:7.2f}")
    
    def _build_extrapolation_model(self):
        """Build cubic spline for smooth interpolation and extrapolation."""
        # Convert dates to numeric values for interpolation
        self.dates_numeric = np.array([
            (d - self.min_date).days for d in self.df['Date']
        ])
        
        # Create cubic spline with data
        self.spline = CubicSpline(
            self.dates_numeric, 
            self.df['Price'].values,
            bc_type='natural'  # Natural spline for smooth extrapolation
        )
        
        # Calculate extrapolation parameters
        # Use last 6 months to estimate trend direction
        last_6_months = self.df.tail(6)
        x_vals = self.dates_numeric[-6:]
        y_vals = last_6_months['Price'].values
        
        # Linear regression for trend
        coeffs = np.polyfit(x_vals, y_vals, 1)
        self.trend_slope = coeffs[0]  # Slope per day
        self.trend_intercept = coeffs[1]
    
    def get_price(self, date):
        """
        Get price estimate for a given date.
        
        For historical dates (within data range): uses interpolation
        For future dates (beyond data range): uses extrapolation with seasonal adjustment
        
        Args:
            date (str or datetime): Date in format 'YYYY-MM-DD' or datetime object
            
        Returns:
            float: Estimated price for the given date
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Calculate days from minimum date
        days_from_start = (date - self.min_date).days
        
        # Check if date is within historical data range
        if self.min_date <= date <= self.max_date:
            # Use spline interpolation for historical dates
            return float(self.spline(days_from_start))
        
        elif date <= self.extrapolation_end:
            # Extrapolate for future dates
            return self._extrapolate_price(date, days_from_start)
        
        else:
            raise ValueError(
                f"Date {date.date()} is beyond extrapolation range "
                f"(max: {self.extrapolation_end.date()})"
            )
    
    def _extrapolate_price(self, date, days_from_start):
        """
        Extrapolate price for future dates using trend + seasonality.
        
        Args:
            date: datetime object
            days_from_start: Number of days from minimum date
            
        Returns:
            float: Extrapolated price
        """
        # Trend component: linear extrapolation of recent trend
        last_day_numeric = self.dates_numeric[-1]
        days_beyond = days_from_start - last_day_numeric
        
        trend_component = (
            self.df['Price'].iloc[-1] + 
            self.trend_slope * days_beyond
        )
        
        # Seasonal component: repeat seasonal pattern
        month = date.month
        seasonal_adjustment = self.seasonal_avg.get(month, 0) - self.seasonal_avg.mean()
        
        # Combine components
        extrapolated_price = trend_component + (seasonal_adjustment * 0.3)
        
        # Floor at a reasonable minimum (natural gas rarely goes negative)
        extrapolated_price = max(extrapolated_price, 0.5)
        
        return float(extrapolated_price)
    
    def get_price_range(self, start_date, end_date, frequency='D'):
        """
        Get prices for a date range.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            frequency (str): Frequency of prices ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            pd.DataFrame: DataFrame with dates and prices
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        prices = [self.get_price(d) for d in date_range]
        
        return pd.DataFrame({
            'Date': date_range,
            'Price': prices
        })
    
    def visualize_analysis(self, save_path=None):
        """
        Create comprehensive visualization of price data and analysis.
        
        Args:
            save_path (str, optional): Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Historical data with spline fit and extrapolation
        ax1 = axes[0, 0]
        ax1.scatter(self.df['Date'], self.df['Price'], color='blue', s=30, 
                   label='Historical Monthly Prices', zorder=3)
        
        # Plot spline fit for historical data
        hist_dates = pd.date_range(self.min_date, self.max_date, freq='D')
        hist_prices = [self.get_price(d) for d in hist_dates]
        ax1.plot(hist_dates, hist_prices, 'b-', alpha=0.5, linewidth=2, label='Spline Interpolation')
        
        # Plot extrapolation
        future_dates = pd.date_range(self.max_date, self.extrapolation_end, freq='D')
        future_prices = [self.get_price(d) for d in future_dates]
        ax1.plot(future_dates, future_prices, 'r--', linewidth=2, label='Extrapolation (1 Year)')
        
        # Add shaded region for extrapolation
        ax1.axvspan(self.max_date, self.extrapolation_end, alpha=0.1, color='red')
        
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Price ($/MMBtu)', fontsize=11)
        ax1.set_title('Natural Gas Price: Historical Data + 1-Year Extrapolation', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Seasonal pattern
        ax2 = axes[0, 1]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_values = [self.seasonal_avg.get(i+1, 0) for i in range(12)]
        colors = ['#d62728' if v > self.seasonal_avg.mean() else '#1f77b4' 
                 for v in seasonal_values]
        ax2.bar(months, seasonal_values, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(self.seasonal_avg.mean(), color='green', linestyle='--', 
                   linewidth=2, label='Annual Average')
        ax2.set_ylabel('Average Price ($/MMBtu)', fontsize=11)
        ax2.set_title('Seasonal Price Pattern by Month', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Trend decomposition
        ax3 = axes[1, 0]
        ax3.plot(self.trend.index, self.trend.values, 'g-', linewidth=2, label='Trend')
        ax3.fill_between(self.trend.index, self.trend.values, alpha=0.3, color='green')
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Trend Component ($/MMBtu)', fontsize=11)
        ax3.set_title('Price Trend Over Time', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Year-over-year comparison
        ax4 = axes[1, 1]
        for year in self.df['Year'].unique():
            year_data = self.df[self.df['Year'] == year]
            ax4.plot(year_data['Month'], year_data['Price'], marker='o', 
                    label=f'{int(year)}', linewidth=2)
        ax4.set_xlabel('Month', fontsize=11)
        ax4.set_ylabel('Price ($/MMBtu)', fontsize=11)
        ax4.set_title('Year-over-Year Price Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(1, 13))
        ax4.set_xticklabels(months)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to {save_path}")
        
        return fig
    
    def storage_contract_analysis(self, injection_date, withdrawal_date):
        """
        Analyze profitability of a storage contract.
        
        Args:
            injection_date (str): Date of purchase/injection (YYYY-MM-DD)
            withdrawal_date (str): Date of sale/withdrawal (YYYY-MM-DD)
            
        Returns:
            dict: Analysis results including costs and potential profit
        """
        injection_price = self.get_price(injection_date)
        withdrawal_price = self.get_price(withdrawal_date)
        
        days_held = (pd.to_datetime(withdrawal_date) - 
                    pd.to_datetime(injection_date)).days
        
        # Simple storage cost estimate (example: $0.50 per MMBtu per month)
        months_held = days_held / 30
        storage_cost = months_held * 0.50
        
        # Calculate spread and profit
        price_spread = withdrawal_price - injection_price
        net_spread = price_spread - storage_cost
        
        return {
            'injection_date': injection_date,
            'injection_price': round(injection_price, 2),
            'withdrawal_date': withdrawal_date,
            'withdrawal_price': round(withdrawal_price, 2),
            'days_held': days_held,
            'price_spread': round(price_spread, 2),
            'estimated_storage_cost': round(storage_cost, 2),
            'net_profit_potential': round(net_spread, 2),
            'profitable': net_spread > 0
        }


def main():
    """Main execution: Load data, build model, and demonstrate functionality."""
    
    # Initialize model
    print("=" * 70)
    print("NATURAL GAS STORAGE CONTRACT PRICING MODEL")
    print("=" * 70)
    
    model = NaturalGasPricingModel(r"C:\Users\grimm\Downloads\FORAGE\JPmorgan\Nat_Gas.csv")
    
    # Example price queries
    print("\n" + "=" * 70)
    print("PRICE ESTIMATION EXAMPLES")
    print("=" * 70)
    
    test_dates = [
        '2020-10-31',  # First historical point
        '2021-06-15',  # Mid-period interpolation
        '2024-09-30',  # Last historical point
        '2024-12-15',  # Near-term extrapolation
        '2025-06-30',  # Mid-term extrapolation
        '2025-09-30',  # End of extrapolation period
    ]
    
    print("\nSample Price Estimates:")
    for test_date in test_dates:
        price = model.get_price(test_date)
        date_obj = pd.to_datetime(test_date)
        if date_obj <= model.max_date:
            data_type = "(Historical)"
        else:
            data_type = "(Extrapolated)"
        print(f"  {test_date} {data_type}: ${price:7.2f}/MMBtu")
    
    # Storage contract example
    print("\n" + "=" * 70)
    print("STORAGE CONTRACT EXAMPLE")
    print("=" * 70)
    
    contract_analysis = model.storage_contract_analysis('2024-06-30', '2024-12-31')
    print("\nSummer-to-Winter Storage Strategy:")
    print(f"  Injection Date:     {contract_analysis['injection_date']}")
    print(f"  Injection Price:    ${contract_analysis['injection_price']}/MMBtu")
    print(f"  Withdrawal Date:    {contract_analysis['withdrawal_date']}")
    print(f"  Withdrawal Price:   ${contract_analysis['withdrawal_price']}/MMBtu")
    print(f"  Days Held:          {contract_analysis['days_held']}")
    print(f"  Price Spread:       ${contract_analysis['price_spread']}/MMBtu")
    print(f"  Storage Cost:       ${contract_analysis['estimated_storage_cost']}/MMBtu")
    print(f"  Net Profit/Loss:    ${contract_analysis['net_profit_potential']}/MMBtu")
    print(f"  Profitable:         {'Yes ✓' if contract_analysis['profitable'] else 'No ✗'}")
    
    # Generate visualization
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    output_folder = r"C:\Users\grimm\Downloads\FORAGE\JPmorgan\outputs"
    os.makedirs(output_folder, exist_ok=True)
    model.visualize_analysis(os.path.join(output_folder, 'natural_gas_analysis.png'))
    
    # Generate price dataset for extended period
    print("\n" + "=" * 70)
    print("GENERATING EXTENDED PRICE DATASET")
    print("=" * 70)
    
    extended_prices = model.get_price_range('2020-10-31', '2025-09-30', frequency='MS')
    extended_prices.to_csv(os.path.join(output_folder, 'extended_nat_gas_prices.csv'), index=False)
    print(f"Extended price data saved ({len(extended_prices)} data points)")
    print("\nFirst few extended prices:")
    print(extended_prices.head(10).to_string(index=False))
    print("\nLast few extended prices:")
    print(extended_prices.tail(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    model = main()
