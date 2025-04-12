import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

age_groups = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']

income_brackets = [
        (0, 25000),
        (25001, 50000), 
        (50001, 100000),
        (100001, 250000),
        (250001, 1000000),
        (1000001, 5000000)  # Added missing bracket
    ]

# 1. Generate synthetic historical market data (unchanged)
def generate_market_data(start_date='2015-01-01', end_date='2023-12-31', frequency='D'):
    """Generate synthetic market data for multiple asset classes"""
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    n_days = len(date_range)
    
    # Base trends with some randomness
    stock_trend = np.cumsum(np.random.normal(0.0005, 0.01, n_days))  # Higher return, higher volatility
    bond_trend = np.cumsum(np.random.normal(0.0002, 0.003, n_days))  # Lower return, lower volatility
    real_estate_trend = np.cumsum(np.random.normal(0.0003, 0.006, n_days))
    cash_trend = np.cumsum(np.random.normal(0.0001, 0.0005, n_days))  # Lowest return, lowest volatility
    
    # Add some correlation between assets
    commodities_trend = 0.7 * stock_trend + 0.3 * np.cumsum(np.random.normal(0.0002, 0.012, n_days))
    crypto_trend = 0.3 * stock_trend + 0.7 * np.cumsum(np.random.normal(0.001, 0.025, n_days))  # Highest volatility
    
    # Add market crashes and recoveries
    # Financial crisis simulation
    crisis_start = int(n_days * 0.2)  # 20% into the timeline
    crisis_duration = 120  # Number of days
    stock_trend[crisis_start:crisis_start+crisis_duration] -= np.linspace(0, 0.4, crisis_duration)
    bond_trend[crisis_start:crisis_start+crisis_duration] -= np.linspace(0, 0.1, crisis_duration)
    real_estate_trend[crisis_start:crisis_start+crisis_duration] -= np.linspace(0, 0.3, crisis_duration)
    commodities_trend[crisis_start:crisis_start+crisis_duration] -= np.linspace(0, 0.35, crisis_duration)
    crypto_trend[crisis_start:crisis_start+crisis_duration] -= np.linspace(0, 0.6, crisis_duration)
    
    # Pandemic simulation
    pandemic_start = int(n_days * 0.7)  # 70% into the timeline
    pandemic_duration = 90  # Number of days
    pandemic_recovery = 200  # Recovery period
    stock_trend[pandemic_start:pandemic_start+pandemic_duration] -= np.linspace(0, 0.35, pandemic_duration)
    stock_trend[pandemic_start+pandemic_duration:pandemic_start+pandemic_duration+pandemic_recovery] += np.linspace(0, 0.45, pandemic_recovery)
    
    # Convert trends to prices (start at reasonable values)
    stock_price = 100 * np.exp(stock_trend)
    bond_price = 100 * np.exp(bond_trend)
    real_estate_price = 100 * np.exp(real_estate_trend)
    cash_value = 100 * np.exp(cash_trend)
    commodities_price = 100 * np.exp(commodities_trend)
    crypto_price = 100 * np.exp(crypto_trend)
    
    # Create DataFrame
    market_data = pd.DataFrame({
        'date': date_range,
        'stock_index': stock_price,
        'bond_index': bond_price,
        'real_estate_index': real_estate_price,
        'commodities_index': commodities_price,
        'crypto_index': crypto_price,
        'cash_index': cash_value
    })
    
    # Add economic indicators
    # Interest rates (correlated negatively with bond prices)
    interest_rates = 0.03 - 0.3 * np.diff(np.log(bond_price), prepend=np.log(bond_price[0])) + np.random.normal(0, 0.0005, n_days)
    interest_rates = np.maximum(0.001, interest_rates)  # Ensure rates are positive
    
    # Inflation (affects all asset classes)
    base_inflation = 0.02 + 0.3 * np.random.normal(0, 0.001, n_days).cumsum()
    inflation_spikes = np.zeros(n_days)
    inflation_spikes[pandemic_start+pandemic_duration:pandemic_start+pandemic_duration+100] = np.linspace(0, 0.04, 100)  # Post-pandemic inflation
    inflation = base_inflation + inflation_spikes
    
    # GDP growth (correlated with stock market)
    gdp_growth_base = 0.025 + 0.4 * np.diff(np.log(stock_price), prepend=np.log(stock_price[0]))
    gdp_growth = gdp_growth_base + np.random.normal(0, 0.002, n_days)
    
    # Unemployment (negatively correlated with stock market)
    unemployment_base = 0.05 - 0.3 * np.diff(np.log(stock_price), prepend=np.log(stock_price[0]))
    unemployment = np.maximum(0.02, unemployment_base + np.random.normal(0, 0.002, n_days))
    unemployment[crisis_start:crisis_start+crisis_duration*2] += np.linspace(0, 0.06, crisis_duration*2)  # Unemployment rises during crisis
    unemployment[pandemic_start:pandemic_start+pandemic_duration*3] += np.linspace(0, 0.08, pandemic_duration*3)  # Unemployment rises during pandemic
    
    # Add economic indicators to the dataframe
    market_data['interest_rate'] = interest_rates
    market_data['inflation_rate'] = inflation
    market_data['gdp_growth_rate'] = gdp_growth
    market_data['unemployment_rate'] = unemployment
    
    return market_data

# 2. Generate income profile data
def generate_income_profiles(num_profiles=1000):
    """Generate synthetic income profiles with demographic information"""
    income_brackets = [
        (0, 25000),
        (25001, 50000), 
        (50001, 100000),
        (100001, 250000),
        (250001, 1000000),
        (1000001, 5000000)  # Added missing bracket
    ]
    
    age_groups = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    education_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
    job_sectors = ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail', 
                  'Government', 'Entertainment', 'Construction', 'Agriculture']
    
    # Generate profiles
    profiles = []
    for i in range(num_profiles):
        # Income (biased toward lower and middle incomes)
        bracket_weights = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]
        income_bracket = random.choices(income_brackets, weights=bracket_weights)[0]
        income = random.randint(income_bracket[0], income_bracket[1])
        
        # Age (correlated with income to some extent)
        income_percentile = sum(w for w in bracket_weights[:income_brackets.index(income_bracket)])
        if income_percentile < 0.3:  # Lower income more likely younger or older
            age_weights = [0.25, 0.15, 0.1, 0.1, 0.15, 0.25]
        elif income_percentile < 0.75:  # Middle income more spread out
            age_weights = [0.1, 0.2, 0.25, 0.25, 0.15, 0.05]
        else:  # Higher income more likely middle-aged to older
            age_weights = [0.05, 0.1, 0.2, 0.3, 0.25, 0.1]
        
        age_group = random.choices(age_groups, weights=age_weights)[0]
        
        # Education (correlated with income)
        if income_percentile < 0.3:
            edu_weights = [0.5, 0.3, 0.15, 0.04, 0.01]
        elif income_percentile < 0.75:
            edu_weights = [0.15, 0.25, 0.4, 0.15, 0.05]
        else:
            edu_weights = [0.05, 0.1, 0.3, 0.4, 0.15]
            
        education = random.choices(education_levels, weights=edu_weights)[0]
        
        # Job sector (some correlation with income)
        if income_percentile < 0.3:
            sector_weights = [0.05, 0.1, 0.02, 0.08, 0.15, 0.3, 0.1, 0.05, 0.1, 0.05]
        elif income_percentile < 0.75:
            sector_weights = [0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
        else:
            sector_weights = [0.3, 0.2, 0.25, 0.05, 0.05, 0.02, 0.05, 0.05, 0.02, 0.01]
            
        job_sector = random.choices(job_sectors, weights=sector_weights)[0]
        
        # Risk tolerance (partially correlated with age and income)
        base_risk = random.normalvariate(0.5, 0.15)  # Base risk centered around middle
        
        # Age factor (younger = more risk tolerance)
        age_index = age_groups.index(age_group)
        age_factor = -0.05 * age_index  # Younger adds risk, older reduces
        
        # Income factor (higher income = slightly more risk tolerance due to buffer)
        income_factor = 0.1 * (income_brackets.index(income_bracket) / len(income_brackets))
        
        # Education factor (higher education = slightly more risk tolerance)
        edu_factor = 0.05 * (education_levels.index(education) / len(education_levels))
        
        # Combine factors and clamp between 0 and 1
        risk_tolerance = max(0.1, min(0.9, base_risk + age_factor + income_factor + edu_factor))
        
        # Investment goals
        possible_goals = ['Retirement', 'Education', 'Home Purchase', 'Wealth Building', 'Income Generation']
        goal_weights = [0.3, 0.15, 0.2, 0.25, 0.1]
        primary_goal = random.choices(possible_goals, weights=goal_weights)[0]
        
        # Investment horizon (correlated with age and goal)
        if age_group in ['56-65', '66+']:
            horizon_base = random.randint(5, 15)  # Shorter horizon for older people
        else:
            horizon_base = random.randint(10, 30)  # Longer horizon for younger people
            
        # Adjust horizon based on goal
        if primary_goal == 'Retirement':
            if age_group == '18-25': horizon_adj = random.randint(35, 45)
            elif age_group == '26-35': horizon_adj = random.randint(25, 35)
            elif age_group == '36-45': horizon_adj = random.randint(15, 25)
            elif age_group == '46-55': horizon_adj = random.randint(10, 20)
            elif age_group == '56-65': horizon_adj = random.randint(5, 15)
            else: horizon_adj = random.randint(3, 10)
        elif primary_goal == 'Education':
            horizon_adj = random.randint(5, 18)  # Shorter timeline
        elif primary_goal == 'Home Purchase':
            horizon_adj = random.randint(3, 10)  # Shorter timeline
        else:
            horizon_adj = horizon_base
            
        investment_horizon = horizon_adj
        
        # Liquidity needs (correlated inversely with income)
        liquidity_base = random.normalvariate(0.5, 0.15)
        income_liquidity_factor = -0.3 * (income_brackets.index(income_bracket) / len(income_brackets))
        liquidity_needs = max(0.1, min(0.9, liquidity_base + income_liquidity_factor))
        
        # Generate profile
        profile = {
            'profile_id': i + 1,
            'age_group': age_group,
            'income': income,
            'income_bracket': f"{income_bracket[0]}-{income_bracket[1]}",
            'education': education,
            'job_sector': job_sector,
            'risk_tolerance': round(risk_tolerance, 2),
            'investment_goal': primary_goal,
            'investment_horizon': investment_horizon,
            'liquidity_needs': round(liquidity_needs, 2)
        }
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

# 3. Generate investment portfolio performance data
def generate_portfolio_performance(market_data, num_portfolios=50, start_date='2018-01-01', end_date='2022-12-31'):
    """Generate synthetic portfolio performance data based on market data"""
    
    # Define some portfolio archetypes
    portfolio_archetypes = [
        # (stocks, bonds, real_estate, commodities, crypto, cash, name, risk_level)
        (0.05, 0.15, 0.05, 0.00, 0.00, 0.75, 'Ultra Conservative', 'Very Low'),
        (0.20, 0.50, 0.10, 0.00, 0.00, 0.20, 'Conservative', 'Low'),
        (0.40, 0.40, 0.10, 0.05, 0.00, 0.05, 'Moderate', 'Medium'),
        (0.60, 0.25, 0.10, 0.05, 0.00, 0.00, 'Growth', 'Medium-High'),
        (0.70, 0.10, 0.10, 0.05, 0.05, 0.00, 'Aggressive Growth', 'High'),
        (0.80, 0.00, 0.05, 0.05, 0.10, 0.00, 'Speculative', 'Very High')
    ]
    
    # Filter market data to desired date range
    filtered_market_data = market_data[(market_data['date'] >= start_date) & (market_data['date'] <= end_date)]
    
    # Create profiles for portfolios
    portfolios = []
    
    for i in range(num_portfolios):
        # Choose a base archetype
        archetype = random.choice(portfolio_archetypes)
        
        # Add some randomization to allocation while preserving sum = 1.0
        stocks = max(0, min(1, archetype[0] + random.normalvariate(0, 0.05)))
        bonds = max(0, min(1, archetype[1] + random.normalvariate(0, 0.05)))
        real_estate = max(0, min(1, archetype[2] + random.normalvariate(0, 0.03)))
        commodities = max(0, min(1, archetype[3] + random.normalvariate(0, 0.02)))
        crypto = max(0, min(1, archetype[4] + random.normalvariate(0, 0.01)))
        
        # Normalize to ensure sum = 1.0
        total = stocks + bonds + real_estate + commodities + crypto
        if total > 0:
            stocks /= total
            bonds /= total
            real_estate /= total
            commodities /= total
            crypto /= total
            cash = 0
        else:
            # Fallback to cash if something went wrong
            cash = 1.0
            stocks = bonds = real_estate = commodities = crypto = 0
        
        # Create portfolio metadata
        portfolio = {
            'portfolio_id': i + 1,
            'name': f"Portfolio {i+1}: {archetype[6]} Variant",
            'risk_level': archetype[7],
            'stocks_allocation': round(stocks, 4),
            'bonds_allocation': round(bonds, 4),
            'real_estate_allocation': round(real_estate, 4),
            'commodities_allocation': round(commodities, 4),
            'crypto_allocation': round(crypto, 4),
            'cash_allocation': round(cash, 4)
        }
        
        portfolios.append(portfolio)
    
    # Calculate performance for each portfolio
    portfolio_performance = []
    
    # Start with $10,000 investment
    initial_investment = 10000
    
    for portfolio in portfolios:
        # Extract allocations
        allocations = {
            'stock_index': portfolio['stocks_allocation'],
            'bond_index': portfolio['bonds_allocation'],
            'real_estate_index': portfolio['real_estate_allocation'],
            'commodities_index': portfolio['commodities_allocation'],
            'crypto_index': portfolio['crypto_allocation'],
            'cash_index': portfolio['cash_allocation']
        }
        
        # Calculate portfolio value for each date
        portfolio_values = []
        
        # Get first row values for normalization
        first_row = filtered_market_data.iloc[0]
        base_values = {col: first_row[col] for col in allocations.keys()}
        
        for _, row in filtered_market_data.iterrows():
            # Calculate weighted sum based on normalized values and allocations
            portfolio_value = sum(
                allocations[asset] * initial_investment * (row[asset] / base_values[asset])
                for asset in allocations.keys()
            )
            
            # Add some random noise (implementation errors, fees, etc.)
            portfolio_value *= (1 + random.normalvariate(0, 0.001))
            
            portfolio_values.append({
                'portfolio_id': portfolio['portfolio_id'],
                'date': row['date'],
                'value': portfolio_value
            })
        
        portfolio_performance.extend(portfolio_values)
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(portfolio_performance)
    portfolios_df = pd.DataFrame(portfolios)
    
    return portfolios_df, performance_df

# 4. Generate optimal portfolio allocation recommendations for income profiles
def generate_portfolio_recommendations(income_profiles, portfolios, portfolio_performance):
    """Generate synthetic mapping between income profiles and portfolio recommendations"""
    
    # Prepare data structures
    recommendations = []
    
    # Get latest performance data for each portfolio
    latest_performance = portfolio_performance.sort_values('date').groupby('portfolio_id').last().reset_index()
    
    # Calculate returns
    portfolio_returns = {}
    for pid in latest_performance['portfolio_id']:
        portfolio_data = portfolio_performance[portfolio_performance['portfolio_id'] == pid]
        initial_value = portfolio_data.iloc[0]['value']
        final_value = portfolio_data.iloc[-1]['value']
        duration_years = (portfolio_data.iloc[-1]['date'] - portfolio_data.iloc[0]['date']).days / 365.25
        
        # Calculate annualized return
        annualized_return = (final_value / initial_value) ** (1 / duration_years) - 1
        
        # Calculate volatility (standard deviation of returns)
        daily_returns = portfolio_data['value'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5)  # Annualized
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        portfolio_returns[pid] = {
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    # For each income profile, assign optimal portfolios
    for _, profile in income_profiles.iterrows():
        # Create decision factors based on profile
        age_factor = 1.0 - 0.15 * age_groups.index(profile['age_group']) / len(age_groups)
        income_bracket_idx = income_brackets.index((int(profile['income_bracket'].split('-')[0]), 
                                                  int(profile['income_bracket'].split('-')[1])))
        income_factor = 0.5 + 0.1 * income_bracket_idx / len(income_brackets)
        
        # Combine factors for risk calculation
        combined_risk_factor = (
            0.3 * profile['risk_tolerance'] + 
            0.3 * age_factor + 
            0.2 * income_factor - 
            0.2 * profile['liquidity_needs']
        )
        
        # Define what makes a good portfolio for this profile
        if profile['investment_goal'] == 'Retirement':
            # Long-term focus
            horizon_weight = 0.5
            return_weight = 0.3
            sharpe_weight = 0.2
        elif profile['investment_goal'] == 'Education' or profile['investment_goal'] == 'Home Purchase':
            # Medium-term, defined goal
            horizon_weight = 0.7
            return_weight = 0.2
            sharpe_weight = 0.1
        else:
            # General wealth building
            horizon_weight = 0.3
            return_weight = 0.4
            sharpe_weight = 0.3
            
        # Score each portfolio
        portfolio_scores = {}
        for pid, p_returns in portfolio_returns.items():
            portfolio_data = portfolios[portfolios['portfolio_id'] == pid].iloc[0]
            
            # Map risk levels to numeric values
            risk_level_map = {
                'Very Low': 0.1,
                'Low': 0.3,
                'Medium': 0.5,
                'Medium-High': 0.7,
                'High': 0.85,
                'Very High': 1.0
            }
            
            portfolio_risk = risk_level_map[portfolio_data['risk_level']]
            
            # Calculate risk match (closer to 0 is better)
            risk_match = 1 - abs(combined_risk_factor - portfolio_risk)
            
            # Calculate horizon match
            if profile['investment_horizon'] < 10:
                # Short horizon - prefer less volatile
                horizon_match = 1 - portfolio_risk
            elif profile['investment_horizon'] < 20:
                # Medium horizon - prefer balanced
                horizon_match = 1 - abs(0.5 - portfolio_risk)
            else:
                # Long horizon - can take more risk
                horizon_match = portfolio_risk
                
            # Calculate return score
            return_score = p_returns['annualized_return'] * 10  # Scale up returns
                
            # Calculate final score
            final_score = (
                0.4 * risk_match + 
                horizon_weight * horizon_match + 
                return_weight * return_score + 
                sharpe_weight * p_returns['sharpe_ratio']
            )
            
            portfolio_scores[pid] = final_score
            
        # Find top 3 portfolios
        top_portfolios = sorted(portfolio_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Create recommendation records
        for rank, (pid, score) in enumerate(top_portfolios):
            recommendation = {
                'profile_id': profile['profile_id'],
                'portfolio_id': pid,
                'rank': rank + 1,
                'match_score': round(score, 4),
                'reasoning': get_recommendation_reasoning(profile, portfolios[portfolios['portfolio_id'] == pid].iloc[0])
            }
            recommendations.append(recommendation)
    
    return pd.DataFrame(recommendations)

# Helper function for generating recommendation reasoning
def get_recommendation_reasoning(profile, portfolio):
    """Generate explanation text for why a portfolio matches an income profile"""
    
    # Templates for different scenarios
    conservative_template = "This {risk_level} risk portfolio is recommended due to {profile_name}'s {age_group} age, {income_bracket} income, and {liquidity} liquidity needs. The {allocation} allocation provides stability while still allowing for some growth over their {horizon}-year investment horizon."
    
    balanced_template = "For {profile_name} in the {age_group} age range with {income_bracket} income, this {risk_level} risk portfolio offers a balanced approach with {allocation} that supports their {goal} goal over a {horizon}-year timeframe. Their {risk_tolerance} risk tolerance and {liquidity} liquidity needs are well-addressed."
    
    aggressive_template = "Given {profile_name}'s {age_group} age, {income_bracket} income, and {risk_tolerance} risk tolerance, this {risk_level} risk portfolio with {allocation} is positioned for strong growth. With a {horizon}-year investment horizon for {goal}, the higher volatility is acceptable given the potential for greater returns."
    
    # Select template based on portfolio risk
    if portfolio['risk_level'] in ['Very Low', 'Low']:
        template = conservative_template
    elif portfolio['risk_level'] in ['Medium', 'Medium-High']:
        template = balanced_template
    else:
        template = aggressive_template
        
    # Get main allocation component
    max_alloc = max(
        portfolio['stocks_allocation'], 
        portfolio['bonds_allocation'],
        portfolio['real_estate_allocation'],
        portfolio['commodities_allocation'],
        portfolio['crypto_allocation'],
        portfolio['cash_allocation']
    )
    
    if max_alloc == portfolio['stocks_allocation']:
        main_alloc = "stock-focused"
    elif max_alloc == portfolio['bonds_allocation']:
        main_alloc = "bond-focused"
    elif max_alloc == portfolio['real_estate_allocation']:
        main_alloc = "real estate-focused"
    elif max_alloc == portfolio['commodities_allocation']:
        main_alloc = "commodities-focused"
    elif max_alloc == portfolio['crypto_allocation']:
        main_alloc = "crypto-focused"
    else:
        main_alloc = "cash-focused"
    
    # Format values for template
    values = {
        'profile_name': f"Profile {profile['profile_id']}",
        'age_group': profile['age_group'],
        'income_bracket': profile['income_bracket'],
        'risk_tolerance': "high" if profile['risk_tolerance'] > 0.7 else "moderate" if profile['risk_tolerance'] > 0.4 else "low",
        'liquidity': "high" if profile['liquidity_needs'] > 0.7 else "moderate" if profile['liquidity_needs'] > 0.4 else "low",
        'risk_level': portfolio['risk_level'],
        'allocation': main_alloc,
        'goal': profile['investment_goal'],
        'horizon': profile['investment_horizon']
    }
    
    return template.format(**values)

# Generate all datasets
def generate_all_datasets(output_dir='.'):
    """Generate all datasets and save to CSV files"""
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Generate market data
    print("Generating market data...")
    market_data = generate_market_data()
    market_data.to_csv(f"{output_dir}/market_data.csv", index=False)
    print(f"Market data saved to {output_dir}/market_data.csv")
    
    # 2. Generate income profiles
    print("Generating income profiles...")
    income_profiles = generate_income_profiles(num_profiles=1000)
    income_profiles.to_csv(f"{output_dir}/income_profiles.csv", index=False)
    print(f"Income profiles saved to {output_dir}/income_profiles.csv")
    
    # 3. Generate portfolio data
    print("Generating portfolio data...")
    portfolios, performance = generate_portfolio_performance(market_data, num_portfolios=50)
    portfolios.to_csv(f"{output_dir}/portfolio_metadata.csv", index=False)
    performance.to_csv(f"{output_dir}/portfolio_performance.csv", index=False)
    print(f"Portfolio data saved to {output_dir}/portfolio_metadata.csv and {output_dir}/portfolio_performance.csv")
    
    # 4. Generate recommendations
    print("Generating portfolio recommendations...")
    recommendations = generate_portfolio_recommendations(income_profiles, portfolios, performance)
    recommendations.to_csv(f"{output_dir}/portfolio_recommendations.csv", index=False)
    print(f"Recommendations saved to {output_dir}/portfolio_recommendations.csv")
    
    # Create sample data for CNN
    print("Creating sample data for CNN...")
    
    # Create features from income profiles, with economic indicators
    # For simplicity, use most recent economic data for each profile
    latest_economic = market_data.iloc[-1][['interest_rate', 'inflation_rate', 'gdp_growth_rate', 'unemployment_rate']]
    
    # Map categorical variables to numeric
    income_profiles_numeric = income_profiles.copy()
    
    # Age group mapping
    age_mapping = {'18-25': 0, '26-35': 1, '36-45': 2, '46-55': 3, '56-65': 4, '66+': 5}
    income_profiles_numeric['age_numeric'] = income_profiles['age_group'].map(age_mapping)
    
    # Education mapping
    edu_mapping = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    income_profiles_numeric['education_numeric'] = income_profiles['education'].map(edu_mapping)
    
    # Job sector mapping
    sectors = ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail', 
                'Government', 'Entertainment', 'Construction', 'Agriculture']
    sector_mapping = {sector: i for i, sector in enumerate(sectors)}
    income_profiles_numeric['job_sector_numeric'] = income_profiles['job_sector'].map(sector_mapping)
    
    # Investment goal mapping
    goals = ['Retirement', 'Education', 'Home Purchase', 'Wealth Building', 'Income Generation']
    goal_mapping = {goal: i for i, goal in enumerate(goals)}
    income_profiles_numeric['investment_goal_numeric'] = income_profiles['investment_goal'].map(goal_mapping)
    
    # Create numeric features dataset
    features = income_profiles_numeric[[
        'profile_id', 'age_numeric', 'income', 'education_numeric', 
        'job_sector_numeric', 'risk_tolerance', 'investment_goal_numeric', 
        'investment_horizon', 'liquidity_needs'
    ]].copy()
    
    # Add economic indicators to each profile
    for col in ['interest_rate', 'inflation_rate', 'gdp_growth_rate', 'unemployment_rate']:
        features[col] = latest_economic[col]
        
    # Get target labels from recommendations
    # For each profile, get the top recommended portfolio's allocations
    cnn_targets = []
    
    for pid in features['profile_id']:
        profile_recommendations = recommendations[recommendations['profile_id'] == pid]
        if len(profile_recommendations) > 0:
            # Get top recommendation
            top_portfolio_id = profile_recommendations.sort_values('rank').iloc[0]['portfolio_id']
            top_portfolio = portfolios[portfolios['portfolio_id'] == top_portfolio_id].iloc[0]
            
            # Extract allocations as target
            target = {
                'profile_id': pid,
                'stocks_allocation': top_portfolio['stocks_allocation'],
                'bonds_allocation': top_portfolio['bonds_allocation'],
                'real_estate_allocation': top_portfolio['real_estate_allocation'],
                'commodities_allocation': top_portfolio['commodities_allocation'],
                'crypto_allocation': top_portfolio['crypto_allocation'],
                'cash_allocation': top_portfolio['cash_allocation'],
                'risk_level': top_portfolio['risk_level']
            }
            cnn_targets.append(target)
    
    cnn_targets_df = pd.DataFrame(cnn_targets)
    
    # Save for CNN
    features.to_csv(f"{output_dir}/cnn_features.csv", index=False)
    cnn_targets_df.to_csv(f"{output_dir}/cnn_targets.csv", index=False)
    print(f"CNN data saved to {output_dir}/cnn_features.csv and {output_dir}/cnn_targets.csv")
    
    # Generate time series data for CNN
    # For each profile, create a series of market conditions and recommended allocations
    print("Generating time series data for CNN...")
    
    # Use a sliding window approach on market data
    window_size = 90  # 90 days of market data
    stride = 30  # Move forward 30 days at a time
    
    # For simplicity, generate time series for a subset of profiles
    sample_profiles = income_profiles.sample(n=100, random_state=42)
    
    time_series_features = []
    time_series_targets = []
    
    for i, window_start in enumerate(range(0, len(market_data) - window_size, stride)):
        # Skip if we don't have enough data for prediction
        if window_start + window_size + 30 >= len(market_data):
            continue
            
        # Get market data for this window
        window_data = market_data.iloc[window_start:window_start+window_size]
        
        # Get future market data (for determining optimal allocation)
        future_data = market_data.iloc[window_start+window_size:window_start+window_size+30]
        
        # For each sample profile
        for _, profile in sample_profiles.iterrows():
            # Calculate optimal allocation based on profile and future performance
            optimal_allocation = calculate_optimal_allocation(profile, future_data)
            
            # Create feature vector
            feature_vector = {
                'profile_id': profile['profile_id'],
                'window_id': i,
                'age_numeric': age_mapping[profile['age_group']],
                'income': profile['income'],
                'education_numeric': edu_mapping[profile['education']],
                'job_sector_numeric': sector_mapping[profile['job_sector']],
                'risk_tolerance': profile['risk_tolerance'],
                'investment_goal_numeric': goal_mapping[profile['investment_goal']],
                'investment_horizon': profile['investment_horizon'],
                'liquidity_needs': profile['liquidity_needs']
            }
            
            # Add market data features (simplify by using summary statistics)
            stock_returns = window_data['stock_index'].pct_change().dropna()
            bond_returns = window_data['bond_index'].pct_change().dropna()
            
            feature_vector.update({
                'stock_mean_return': stock_returns.mean(),
                'stock_volatility': stock_returns.std(),
                'bond_mean_return': bond_returns.mean(),
                'bond_volatility': bond_returns.std(),
                'interest_rate_mean': window_data['interest_rate'].mean(),
                'inflation_rate_mean': window_data['inflation_rate'].mean(),
                'unemployment_rate_mean': window_data['unemployment_rate'].mean(),
            })
            
            time_series_features.append(feature_vector)
            time_series_targets.append(optimal_allocation)
    
    # Convert to DataFrames
    ts_features_df = pd.DataFrame(time_series_features)
    ts_targets_df = pd.DataFrame(time_series_targets)
    
    # Save time series data
    ts_features_df.to_csv(f"{output_dir}/time_series_features.csv", index=False)
    ts_targets_df.to_csv(f"{output_dir}/time_series_targets.csv", index=False)
    print(f"Time series data saved to {output_dir}/time_series_features.csv and {output_dir}/time_series_targets.csv")
    
    print("All datasets generated successfully!")
    
    return {
        'market_data': market_data,
        'income_profiles': income_profiles,
        'portfolios': portfolios,
        'performance': performance,
        'recommendations': recommendations,
        'cnn_features': features,
        'cnn_targets': cnn_targets_df,
        'ts_features': ts_features_df,
        'ts_targets': ts_targets_df
    }

# Helper function for calculating optimal allocation
def calculate_optimal_allocation(profile, future_market_data):
    """Calculate the optimal allocation for a profile based on future market data"""
    
    # Define allocation archetypes
    allocation_archetypes = [
        # (stocks, bonds, real_estate, commodities, crypto, cash, risk_level)
        (0.05, 0.15, 0.05, 0.00, 0.00, 0.75, 'Very Low'),
        (0.20, 0.50, 0.10, 0.00, 0.00, 0.20, 'Low'),
        (0.40, 0.40, 0.10, 0.05, 0.00, 0.05, 'Medium'),
        (0.60, 0.25, 0.10, 0.05, 0.00, 0.00, 'Medium-High'),
        (0.70, 0.10, 0.10, 0.05, 0.05, 0.00, 'High'),
        (0.80, 0.00, 0.05, 0.05, 0.10, 0.00, 'Very High')
    ]
    
    # Calculate returns for each asset class
    first_row = future_market_data.iloc[0]
    last_row = future_market_data.iloc[-1]
    
    returns = {
        'stock_index': last_row['stock_index'] / first_row['stock_index'] - 1,
        'bond_index': last_row['bond_index'] / first_row['bond_index'] - 1,
        'real_estate_index': last_row['real_estate_index'] / first_row['real_estate_index'] - 1,
        'commodities_index': last_row['commodities_index'] / first_row['commodities_index'] - 1,
        'crypto_index': last_row['crypto_index'] / first_row['crypto_index'] - 1,
        'cash_index': last_row['cash_index'] / first_row['cash_index'] - 1
    }
    
    # Calculate performance for each archetype
    archetype_performance = []
    for stocks, bonds, real_estate, commodities, crypto, cash, risk_level in allocation_archetypes:
        # Calculate return
        portfolio_return = (
            stocks * returns['stock_index'] +
            bonds * returns['bond_index'] +
            real_estate * returns['real_estate_index'] +
            commodities * returns['commodities_index'] +
            crypto * returns['crypto_index'] +
            cash * returns['cash_index']
        )
        
        # Create a score based on return and profile characteristics
        
        # Risk preference factor
        risk_map = {
            'Very Low': 0.1,
            'Low': 0.3,
            'Medium': 0.5,
            'Medium-High': 0.7,
            'High': 0.85,
            'Very High': 1.0
        }
        
        risk_score = risk_map[risk_level]
        risk_preference = profile['risk_tolerance']
        
        # Risk match (1.0 means perfect match)
        risk_match = 1.0 - abs(risk_preference - risk_score)
        
        # Horizon factor (longer horizon can take more risk)
        horizon_factor = min(1.0, profile['investment_horizon'] / 30)
        
        # Liquidity factor (higher liquidity needs prefer lower risk)
        liquidity_factor = 1.0 - profile['liquidity_needs']
        
        # Calculate score
        # Weight factors based on importance
        score = (
            0.4 * portfolio_return +
            0.3 * risk_match +
            0.2 * (risk_score * horizon_factor) +
            0.1 * (1.0 - risk_score * profile['liquidity_needs'])
        )
        
        archetype_performance.append((stocks, bonds, real_estate, commodities, crypto, cash, risk_level, score))
    
    # Sort by score and take the best
    best_archetype = sorted(archetype_performance, key=lambda x: x[7], reverse=True)[0]
    
    # Return optimal allocation
    return {
        'stocks_allocation': best_archetype[0],
        'bonds_allocation': best_archetype[1],
        'real_estate_allocation': best_archetype[2],
        'commodities_allocation': best_archetype[3],
        'crypto_allocation': best_archetype[4],
        'cash_allocation': best_archetype[5],
        'risk_level': best_archetype[6]
    }

# Example usage
if __name__ == "__main__":
    datasets = generate_all_datasets()
    
    # Print shapes of generated datasets
    for name, df in datasets.items():
        print(f"{name}: {df.shape}")