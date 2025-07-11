
"""
Options Pricing Validation Script
Compare our implementation with professional libraries
"""


import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from src.models.option_pricing import BlackScholesCalculator
except ImportError:
    print("Error: Could not import BlackScholesCalculator")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def validate_with_external_libraries():
    """
    Compare our Black-Scholes implementation with external libraries
    """
    # Test parameters (same as before)
    S = 24000      # Spot price
    K = 24000      # Strike price  
    T = 30/365     # Time to expiry (30 days)
    r = 0.065      # Risk-free rate (6.5%)
    sigma = 0.15   # Volatility (15%)

    print("🔍 VALIDATION: Comparing Pricing Models")
    print("=" * 60)
    print(f"Parameters: S=₹{S}, K=₹{K}, T={T*365:.0f}days, r={r:.1%}, σ={sigma:.1%}")
    print()

    try:
        # Our implementation
        our_calc = BlackScholesCalculator(S, K, T, r, sigma)
        our_call_price = our_calc.call_price()
        our_put_price = our_calc.put_price()
        our_call_delta = our_calc.delta('call')
        our_put_delta = our_calc.delta('put')
        
        # Validate put-call parity
        parity_valid = our_calc.validate_put_call_parity()
        
    except Exception as e:
        print(f"❌ Error in our implementation: {e}")
        return

    print("📊 OUR IMPLEMENTATION:")
    print(f"  Call Price: ₹{our_call_price:.2f}")
    print(f"  Put Price:  ₹{our_put_price:.2f}")
    print(f"  Call Delta: {our_call_delta:.4f}")
    print(f"  Put Delta:  {our_put_delta:.4f}")
    print(f"  Put-Call Parity Valid: {'✅' if parity_valid else '❌'}")
    print()

    # Try mibian library
    try:
        import mibian

        # mibian expects: [S, K, r, T_days], volatility=sigma*100
        mibian_bs = mibian.BS([S, K, r*100, T*365], volatility=sigma*100)

        print("📊 MIBIAN LIBRARY:")
        print(f"  Call Price: ₹{mibian_bs.callPrice:.2f}")
        print(f"  Put Price:  ₹{mibian_bs.putPrice:.2f}")
        print(f"  Call Delta: {mibian_bs.callDelta:.4f}")
        print(f"  Put Delta:  {mibian_bs.putDelta:.4f}")
        print()

        # Calculate differences
        call_diff = abs(our_call_price - mibian_bs.callPrice)
        put_diff = abs(our_put_price - mibian_bs.putPrice)

        print("✅ COMPARISON WITH MIBIAN:")
        print(f"  Call Price Difference: ₹{call_diff:.2f}")
        print(f"  Put Price Difference:  ₹{put_diff:.2f}")

        if call_diff < 1 and put_diff < 1:
            print("  🎉 Excellent! Our implementation matches mibian closely!")
        else:
            print("  ⚠️  Some differences found - need investigation")
        print()

    except ImportError:
        print("❌ Mibian library not installed")
        print("   Install with: pip install mibian")
        print()

    # Try py_vollib library
    try:
        from py_vollib.black_scholes import black_scholes
        from py_vollib.black_scholes.greeks.analytical import delta

        # py_vollib expects: flag, S, K, T, r, sigma
        vollib_call_price = black_scholes('c', S, K, T, r, sigma)
        vollib_put_price = black_scholes('p', S, K, T, r, sigma)
        vollib_call_delta = delta('c', S, K, T, r, sigma)
        vollib_put_delta = delta('p', S, K, T, r, sigma)

        print("📊 PY_VOLLIB LIBRARY:")
        print(f"  Call Price: ₹{vollib_call_price:.2f}")
        print(f"  Put Price:  ₹{vollib_put_price:.2f}")
        print(f"  Call Delta: {vollib_call_delta:.4f}")
        print(f"  Put Delta:  {vollib_put_delta:.4f}")
        print()

        # Calculate differences
        call_diff = abs(our_call_price - vollib_call_price)
        put_diff = abs(our_put_price - vollib_put_price)

        print("✅ COMPARISON WITH PY_VOLLIB:")
        print(f"  Call Price Difference: ₹{call_diff:.2f}")
        print(f"  Put Price Difference:  ₹{put_diff:.2f}")

        if call_diff < 1 and put_diff < 1:
            print("  🎉 Excellent! Our implementation matches py_vollib closely!")
        else:
            print("  ⚠️  Some differences found - need investigation")

    except ImportError:
        print("❌ py_vollib library not installed")
        print("   Install with: pip install py-vollib")
        print()

def test_with_real_market_data():
    """
    Test our pricing model with realistic Indian market scenarios
    """
    print("🇮🇳 TESTING WITH INDIAN MARKET SCENARIOS")
    print("=" * 60)

    scenarios = [
        {
            'name': 'NIFTY ATM Weekly',
            'spot': 24000, 'strike': 24000, 'days': 7, 
            'rate': 0.065, 'vol': 0.12, 'description': 'At-the-money weekly option'
        },
        {
            'name': 'NIFTY OTM Monthly', 
            'spot': 24000, 'strike': 25000, 'days': 30,
            'rate': 0.065, 'vol': 0.15, 'description': 'Out-of-the-money monthly option'
        },
        {
            'name': 'BANKNIFTY High Vol',
            'spot': 51000, 'strike': 51000, 'days': 14,
            'rate': 0.065, 'vol': 0.20, 'description': 'High volatility scenario'
        }
    ]

    for scenario in scenarios:
        print(f"\n📈 {scenario['name']}: {scenario['description']}")

        calc = BlackScholesCalculator(
            spot_price=scenario['spot'],
            strike_price=scenario['strike'], 
            time_to_expiry=scenario['days']/365,
            risk_free_rate=scenario['rate'],
            volatility=scenario['vol']
        )

        call_price = calc.call_price()
        put_price = calc.put_price()
        call_delta = calc.delta('call')

        print(f"  Spot: ₹{scenario['spot']:,} | Strike: ₹{scenario['strike']:,} | {scenario['days']} days")
        print(f"  Call: ₹{call_price:.2f} | Put: ₹{put_price:.2f} | Delta: {call_delta:.3f}")

if __name__ == "__main__":
    validate_with_external_libraries()
    test_with_real_market_data()
