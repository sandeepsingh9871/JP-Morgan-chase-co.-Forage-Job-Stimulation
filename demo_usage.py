"""
NATURAL GAS PRICING MODEL - USAGE GUIDE
========================================

This guide demonstrates how to use the NaturalGasPricingModel class to:
1. Get price estimates for any date
2. Analyze storage contracts
3. Access seasonal insights
"""

import os
from natural_gas_pricing_model import NaturalGasPricingModel
import pandas as pd


def _get_default_csv_path():
    base_dir = os.path.dirname(__file__)
    preferred_paths = [
        os.path.join(base_dir, 'Nat_Gas.csv'),
        os.path.join(base_dir, 'data', 'Nat_Gas.csv'),
    ]
    for path in preferred_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        'Could not find Nat_Gas.csv in the project directory. '
        'Place the file alongside demo_usage.py or in a data/ subfolder.'
    )


def demo_basic_usage(model):
    """Demonstrate basic usage of the pricing model."""
    
    print("\n" + "="*70)
    print("DEMO: BASIC PRICE ESTIMATION")
    print("="*70)
    
    # Get price for a specific date
    date = '2024-12-15'
    price = model.get_price(date)
    print(f"\nNatural gas price on {date}: ${price:.2f}/MMBtu")
    
    # Get prices for a date range (daily)
    print("\nDaily prices for December 2024:")
    dec_2024 = model.get_price_range('2024-12-01', '2024-12-31', frequency='D')
    print(f"  Min: ${dec_2024['Price'].min():.2f}")
    print(f"  Max: ${dec_2024['Price'].max():.2f}")
    print(f"  Avg: ${dec_2024['Price'].mean():.2f}")
    
    # Get monthly prices for extrapolation period
    print("\nMonthly prices for extrapolation period (Oct 2024 - Sep 2025):")
    monthly = model.get_price_range('2024-10-01', '2025-09-30', frequency='MS')
    print(monthly.to_string(index=False))
    
    return model


def demo_storage_contracts(model):
    """Demonstrate storage contract analysis."""
    
    print("\n" + "="*70)
    print("DEMO: STORAGE CONTRACT ANALYSIS")
    print("="*70)
    
    # Example 1: Summer-to-Winter strategy (classic gas storage play)
    print("\n1. SUMMER-TO-WINTER STORAGE (Buy Low, Sell High)")
    print("-" * 70)
    result1 = model.storage_contract_analysis('2024-07-01', '2024-12-31')
    print(f"   Inject natural gas during summer (cheaper): {result1['injection_date']}")
    print(f"   Injection price: ${result1['injection_price']}/MMBtu")
    print(f"   ")
    print(f"   Withdraw gas during winter (more expensive): {result1['withdrawal_date']}")
    print(f"   Withdrawal price: ${result1['withdrawal_price']}/MMBtu")
    print(f"   ")
    print(f"   Days in storage: {result1['days_held']}")
    print(f"   Price spread: ${result1['price_spread']}/MMBtu")
    print(f"   Storage cost: ${result1['estimated_storage_cost']}/MMBtu")
    print(f"   Net profit potential: ${result1['net_profit_potential']}/MMBtu")
    
    # Example 2: Short-term contract
    print("\n2. SHORT-TERM STORAGE (3 months)")
    print("-" * 70)
    result2 = model.storage_contract_analysis('2025-01-01', '2025-04-01')
    print(f"   Injection date: {result2['injection_date']}")
    print(f"   Injection price: ${result2['injection_price']}/MMBtu")
    print(f"   Withdrawal date: {result2['withdrawal_date']}")
    print(f"   Withdrawal price: ${result2['withdrawal_price']}/MMBtu")
    print(f"   Days held: {result2['days_held']}")
    print(f"   Net profit: ${result2['net_profit_potential']}/MMBtu")
    
    # Example 3: Year-long contract
    print("\n3. LONG-TERM STORAGE (10+ months)")
    print("-" * 70)
    result3 = model.storage_contract_analysis('2024-10-15', '2025-09-15')
    print(f"   Injection date: {result3['injection_date']}")
    print(f"   Injection price: ${result3['injection_price']}/MMBtu")
    print(f"   Withdrawal date: {result3['withdrawal_date']}")
    print(f"   Withdrawal price: ${result3['withdrawal_price']}/MMBtu")
    print(f"   Days held: {result3['days_held']}")
    print(f"   Net profit: ${result3['net_profit_potential']}/MMBtu")


def demo_seasonal_insights(model):
    """Demonstrate seasonal pattern analysis."""
    
    print("\n" + "="*70)
    print("DEMO: SEASONAL INSIGHTS")
    print("="*70)
    
    # Identify expensive and cheap months
    seasonal = model.seasonal_avg.sort_values(ascending=False)
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    print("\nMost Expensive Months (Best for SELLING):")
    for i, (month, price) in enumerate(seasonal.head(5).items(), 1):
        print(f"  {i}. {month_names[month]:12s}: ${price:7.2f}/MMBtu")
    
    print("\nCheapest Months (Best for BUYING):")
    for i, (month, price) in enumerate(seasonal.tail(5)[::-1].items(), 1):
        print(f"  {i}. {month_names[month]:12s}: ${price:7.2f}/MMBtu")
    
    # Calculate seasonal spread
    avg_price = model.seasonal_avg.mean()
    max_month = seasonal.idxmax()
    min_month = seasonal.idxmin()
    max_price = seasonal.max()
    min_price = seasonal.min()
    spread = max_price - min_price
    
    print(f"\nSeasonal Analysis:")
    print(f"  Annual average price: ${avg_price:.2f}/MMBtu")
    print(f"  Peak month: {month_names[max_month]} (${max_price:.2f}/MMBtu)")
    print(f"  Trough month: {month_names[min_month]} (${min_price:.2f}/MMBtu)")
    print(f"  Seasonal spread: ${spread:.2f}/MMBtu")
    print(f"  Spread as % of average: {(spread/avg_price)*100:.1f}%")


def demo_advanced_pricing(model):
    """Demonstrate advanced pricing scenarios."""
    
    print("\n" + "="*70)
    print("DEMO: ADVANCED PRICING SCENARIOS")
    print("="*70)
    
    # Generate daily prices for Q1 2025
    print("\nQ1 2025 Daily Prices (Sample):")
    q1_prices = model.get_price_range('2025-01-01', '2025-03-31', frequency='D')
    
    print(f"\n  Total trading days: {len(q1_prices)}")
    print(f"  Min daily price: ${q1_prices['Price'].min():.2f}")
    print(f"  Max daily price: ${q1_prices['Price'].max():.2f}")
    print(f"  Average: ${q1_prices['Price'].mean():.2f}")
    print(f"  Std Dev: ${q1_prices['Price'].std():.2f}")
    
    # Show first week and last week
    print("\n  First week of January:")
    print(q1_prices.head(7)[['Date', 'Price']].to_string(index=False))
    
    print("\n  Last week of March:")
    print(q1_prices.tail(7)[['Date', 'Price']].to_string(index=False))


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  NATURAL GAS PRICING MODEL - DEMONSTRATIONS".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    model = NaturalGasPricingModel(_get_default_csv_path())
    demo_basic_usage(model)
    demo_storage_contracts(model)
    demo_seasonal_insights(model)
    demo_advanced_pricing(model)
    
    print("\n" + "="*70)
    print("DEMONSTRATIONS COMPLETE")
    print("="*70)
    print("\nKey Insights from Data:")
    print("  • Winter months (Jan-Mar) show higher prices")
    print("  • Summer months (Jun-Jul) show lower prices")
    print("  • This creates opportunities for seasonal storage strategies")
    print("  • The model can estimate prices for any date within ±1 year")
    print("\nFor production use, consider:")
    print("  • Adding real-time data feeds")
    print("  • Incorporating weather data (affects demand)")
    print("  • Adding geopolitical risk factors")
    print("  • Implementing more sophisticated forecasting models")


if __name__ == "__main__":
    main()
