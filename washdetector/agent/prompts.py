# washdetector/agent/prompts.py

from typing import List, Dict
from datetime import datetime
from collections import defaultdict

def prepare_transaction_stats(transactions: List[Dict]) -> Dict:
    """Prepare statistical summaries of transactions"""
    stats = {
        'unique_addresses': set(),
        'address_volumes': defaultdict(float),
        'address_partners': defaultdict(set),
        'time_start': float('inf'),
        'time_end': 0,
        'price_ranges': defaultdict(lambda: {'min': float('inf'), 'max': -float('inf')}),
        'volume_per_hour': defaultdict(float)
    }
    
    for tx in transactions:
        from_addr = tx['from_addr']
        to_addr = tx['to_addr']
        amount = tx['token_amount']
        price = tx['price_eth']
        timestamp = tx['timestamp']
        
        stats['unique_addresses'].add(from_addr)
        stats['unique_addresses'].add(to_addr)
        stats['address_volumes'][from_addr] += amount
        stats['address_volumes'][to_addr] += amount
        stats['address_partners'][from_addr].add(to_addr)
        stats['time_start'] = min(stats['time_start'], timestamp)
        stats['time_end'] = max(stats['time_end'], timestamp)
        stats['price_ranges'][from_addr]['min'] = min(stats['price_ranges'][from_addr]['min'], price)
        stats['price_ranges'][from_addr]['max'] = max(stats['price_ranges'][from_addr]['max'], price)
        
        hour = timestamp - (timestamp % 3600)
        stats['volume_per_hour'][hour] += amount
        
    return stats

def create_analysis_prompt(transactions: List[Dict], augmented_features: str = None) -> str:
    """Create comprehensive analysis prompt with chain-of-thought structure"""
    stats = prepare_transaction_stats(transactions)
    
    txs = "\n".join([
        f"Transaction {i+1}:"
        f"\nFrom: {tx['from_addr']}"
        f"\nTo: {tx['to_addr']}"
        f"\nAmount: {tx['token_amount']:.2f}"
        f"\nPrice: {tx['price_eth']:.2f}"
        f"\nTimestamp: {datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}"
        for i, tx in enumerate(transactions[:10])
    ])

    base_prompt = f"""You are a cryptocurrency forensics expert analyzing trading data to detect wash trading. Use the formal definition from academic research and the provided metrics to identify wash trading patterns.

KEY WASH TRADING CHARACTERISTICS:
1. Legal Definition:
- Trades that "give the appearance that purchases and sales have been made, without incurring market risk or changing the trader's market position"
- Actors executing trades where they end up at the same market position they had initially
- Trades between colluding parties with no real change in ownership

2. Important Distinctions:
- High volume alone is NOT indicative of wash trading
- Market makers legitimately create high volumes through normal market-making activities
- Transaction types in data are labeled as: 'normal', 'market_maker', 'wash', 'camouflage'

Dataset Overview:
Total Transactions: {len(transactions)}
Unique Addresses: {len(stats['unique_addresses'])}
Time Period: {datetime.fromtimestamp(stats['time_start']).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(stats['time_end']).strftime('%Y-%m-%d %H:%M:%S')}

Sample Transactions:
{txs}

Augmented Features:
{augmented_features if augmented_features else ""}
///End Augmented Features///

Analyze this data following these steps:

1. Network Analysis:
- Use cycle detection metrics to identify wash trading loops
- Consider volume symmetry scores in your analysis
- Map token flow patterns and circular trading

2. Position Analysis:
- Use position reversion metrics to identify suspicious patterns
- Look for addresses with high symmetry and frequent reversions
- Consider normal market maker behavior

3. Volume Analysis:
- Examine volume concentration ratios
- Look for suspicious volume symmetry patterns
- Consider normal trading variance

Only after completing all analysis steps, provide your final assessment in this format:

DETAILED REASONING:
[Provide your complete chain of thought]

EVIDENCE SUMMARY:
[List key evidence from each analysis step]

WASH TRADING ASSESSMENT:
[Conclude whether wash trading is likely occurring]

CONFIDENCE LEVEL:
[Provide confidence as percentage and explain why]

SUSPICIOUS ADDRESSES:
[List addresses with suspicious patterns]
"""
    return base_prompt