import pandas as pd
import sys
import requests
import numpy as np
import ast
from data_updater.trading_utils import get_clob_client
from data_updater.google_utils import get_spreadsheet
from data_updater.find_markets import get_all_markets, get_bid_ask_range, generate_numbers, add_formula_params, calculate_annualized_volatility
from gspread_dataframe import set_with_dataframe
import traceback

def normalize_question(q):
    # Normalize a string by lowercasing and removing spaces and hyphens
    if isinstance(q, str):
        return q.lower().replace(' ', '').replace('-', '')
    else:
        return q  # For Series, handle in filter

def calculate_annualized_volatility(df, hours):
    end_time = df['t'].max()
    start_time = end_time - pd.Timedelta(hours=hours)
    window_df = df[df['t'] >= start_time]
    volatility = window_df['log_return'].std()
    if pd.isna(volatility) or np.isinf(volatility):
        return 0
    annualized_volatility = volatility * np.sqrt(60 * 24 * 252)
    return round(annualized_volatility, 2)

def add_volatility(row):
    try:
        res = requests.get(f'https://clob.polymarket.com/prices-history?interval=1m&market={row["token1"]}&fidelity=10')
        price_df = pd.DataFrame(res.json()['history'])
        price_df['t'] = pd.to_datetime(price_df['t'], unit='s')
        price_df['p'] = price_df['p'].round(2)
        price_df.to_csv(f'data/{row["token1"]}.csv', index=False)
        price_df['log_return'] = np.log(price_df['p'] / price_df['p'].shift(1))
        row_dict = row.copy()
        stats = {
            '1_hour': calculate_annualized_volatility(price_df, 1),
            '3_hour': calculate_annualized_volatility(price_df, 3),
            '6_hour': calculate_annualized_volatility(price_df, 6),
            '12_hour': calculate_annualized_volatility(price_df, 12),
            '24_hour': calculate_annualized_volatility(price_df, 24),
            '7_day': calculate_annualized_volatility(price_df, 24 * 7),
            '14_day': calculate_annualized_volatility(price_df, 24 * 14),
            '30_day': calculate_annualized_volatility(price_df, 24 * 30),
            'volatility_price': price_df['p'].iloc[-1]
        }
        new_dict = {**row_dict, **stats}
        return new_dict
    except:
        print("Error fetching volatility")
        return row

def process_single_row(row, max_workers=5):
    ret = {}
    ret['question'] = row.get('question', '')
    ret['neg_risk'] = row.get('negRisk', False)

    # Handle Gamma API format
    outcomes = row.get('outcomes', 'Yes,No')
    if isinstance(outcomes, str):
        # Check if it's a JSON-like list string
        if outcomes.startswith('[') and outcomes.endswith(']'):
            try:
                outcomes = ast.literal_eval(outcomes)
            except (ValueError, SyntaxError):
                outcomes = outcomes.split(',')
        else:
            outcomes = outcomes.split(',')
    elif not isinstance(outcomes, list):
        outcomes = ['Yes', 'No']
    ret['answer1'] = str(outcomes[0]) if len(outcomes) > 0 else 'Yes'
    ret['answer2'] = str(outcomes[1]) if len(outcomes) > 1 else 'No'

    ret['min_size'] = row.get('orderMinSize', 1)
    ret['max_spread'] = row.get('rewardsMaxSpread', 10)

    clob_token_ids = row.get('clobTokenIds', '')
    if isinstance(clob_token_ids, str):
        # Check if it's a JSON-like list string
        if clob_token_ids.startswith('[') and clob_token_ids.endswith(']'):
            try:
                clob_token_ids = ast.literal_eval(clob_token_ids)
            except (ValueError, SyntaxError):
                clob_token_ids = clob_token_ids.split(',')
        else:
            clob_token_ids = clob_token_ids.split(',')
    elif not isinstance(clob_token_ids, list):
        clob_token_ids = [None, None]
    token1 = clob_token_ids[0] if len(clob_token_ids) > 0 else None
    token2 = clob_token_ids[1] if len(clob_token_ids) > 1 else None

    # No rewards in Gamma, set to 0
    ret['rewards_daily_rate'] = 0
    ret['bid_reward_per_100'] = 0
    ret['ask_reward_per_100'] = 0
    ret['sm_reward_per_100'] = 0
    ret['gm_reward_per_100'] = 0

    # Use Gamma data for best_bid, best_ask, spread
    ret['best_bid'] = row.get('bestBid', 0)
    ret['best_ask'] = row.get('bestAsk', 0)
    ret['spread'] = abs(ret['best_ask'] - ret['best_bid'])

    ret['midpoint'] = (ret['best_bid'] + ret['best_ask']) / 2

    TICK_SIZE = row.get('orderPriceMinTickSize', 0.01)
    ret['tick_size'] = TICK_SIZE

    # Since no orderbook, set rewards to 0
    best_bid_reward = 0
    best_ask_reward = 0

    ret['bid_reward_per_100'] = best_bid_reward
    ret['ask_reward_per_100'] = best_ask_reward

    ret['sm_reward_per_100'] = round((best_bid_reward + best_ask_reward) / 2, 2)
    ret['gm_reward_per_100'] = round((best_bid_reward * best_ask_reward) ** 0.5, 2)

    ret['end_date_iso'] = row.get('endDateIso', '')
    ret['market_slug'] = row.get('slug', '')
    ret['token1'] = str(token1) if token1 else ''
    ret['token2'] = str(token2) if token2 else ''
    ret['condition_id'] = row.get('conditionId', '')

    return ret

# Initialize global variables
spreadsheet = get_spreadsheet()
client = get_clob_client()

def update_sheet(data, worksheet):
    all_values = worksheet.get_all_values()
    existing_num_rows = len(all_values)
    existing_num_cols = len(all_values[0]) if all_values else 0

    num_rows, num_cols = data.shape
    max_rows = max(num_rows, existing_num_rows)
    max_cols = max(num_cols, existing_num_cols)

    # Create a DataFrame with the maximum size and fill it with empty strings
    padded_data = pd.DataFrame('', index=range(max_rows), columns=range(max_cols))

    # Update the padded DataFrame with the original data and its columns
    padded_data.iloc[:num_rows, :num_cols] = data.values
    padded_data.columns = list(data.columns) + [''] * (max_cols - num_cols)

    # Update the sheet with the padded DataFrame, including column headers
    set_with_dataframe(worksheet, padded_data, include_index=False, include_column_header=True, resize=True)

def main(question):
    try:
        # Get all markets
        all_df = get_all_markets(client)
        print(f"Fetched {len(all_df)} markets.")

        # Filter by question using normalized comparison
        normalized_question = normalize_question(question)
        filtered_df = all_df[all_df['question'].str.lower().str.replace(' ', '').str.replace('-', '') == normalized_question]

        if filtered_df.empty:
            print(f"No markets found for question: {question}")
            print("Normalized question:", normalized_question)
            # print("Sample questions from data:")
            # print(all_df['question'].head(10).tolist())
            # print("Sample normalized questions from data:")
            # print(all_df['question'].str.lower().str.replace(' ', '').str.replace('-', '').head(10).tolist())
            # Try fetching by slug (generate slug by lowercasing and replacing spaces with hyphens)
            slug = question.lower().replace(' ', '-')
            url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
            print(f"Trying to fetch by slug: {slug}")
            res = requests.get(url)
            if res.status_code == 200:
                event = res.json()
                if event.get('markets'):
                    filtered_df = pd.DataFrame(event['markets'])
                    print(f"Found {len(filtered_df)} markets via slug.")
                else:
                    print("No markets in event.")
                    return
            else:
                print(f"Failed to fetch event: {res.status_code}")
                return

        print(f"Found {len(filtered_df)} markets matching the question.")

        # Get detailed results
        all_results = []
        for idx, row in filtered_df.iterrows():
            try:
                result = process_single_row(row.to_dict())
                all_results.append(result)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")

        print("Processed market details.")

        # Create DataFrame
        new_df = pd.DataFrame(all_results)

        if new_df.empty:
            print("No valid market data after processing.")
            return

        # Add spread
        new_df['spread'] = abs(new_df['best_ask'] - new_df['best_bid'])

        # Add volatility
        # Add volatility
        new_df = pd.DataFrame([add_volatility(row.to_dict()) for _, row in new_df.iterrows()])
    
        # Ensure volatility columns exist
        volatility_cols = ['1_hour', '3_hour', '6_hour', '12_hour', '24_hour', '7_day', '14_day', '30_day', 'volatility_price', 'volatility_sum']
        for col in volatility_cols:
            if col not in new_df.columns:
                new_df[col] = 0
    
        new_df['volatility_sum'] = new_df['24_hour'] + new_df['7_day'] + new_df['30_day']

        # Calculate volatilty/reward
        new_df['volatilty/reward'] = ((new_df['gm_reward_per_100'] / new_df['volatility_sum']).round(2)).astype(str)

        # Select columns matching the original script
        new_df = new_df[['question', 'answer1', 'answer2', 'spread', 'rewards_daily_rate', 'gm_reward_per_100', 'sm_reward_per_100', 'bid_reward_per_100', 'ask_reward_per_100', 'volatility_sum', 'volatilty/reward', 'min_size', '1_hour', '3_hour', '6_hour', '12_hour', '24_hour', '7_day', '30_day', 'best_bid', 'best_ask', 'volatility_price', 'max_spread', 'tick_size', 'neg_risk', 'market_slug', 'token1', 'token2', 'condition_id']]

        # Get current Spread Markets data
        wk_spread = spreadsheet.worksheet("Spread Markets")
        current_df = pd.DataFrame(wk_spread.get_all_records())

        # Combine: new markets on top
        combined_df = pd.concat([new_df, current_df], ignore_index=True) if not current_df.empty else new_df
    
        # Replace inf and -inf with 0
        combined_df = combined_df.replace([np.inf, -np.inf], 0)
    
        print(f"Updating Spread Markets with {len(combined_df)} rows.")
    
        # Update the sheet
        update_sheet(combined_df, wk_spread)

        print("Successfully updated Spread Markets.")

    except Exception as e:
        traceback.print_exc()
        print(str(e))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python query_and_add_spread.py '<question>'")
        sys.exit(1)

    question = sys.argv[1]
    main(question)