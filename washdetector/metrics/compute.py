import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
from datetime import datetime

def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Safely compute mean with default value"""
    return float(np.mean(values)) if values else default

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def compute_volume_metrics(transactions: List[Dict]) -> Dict:
    """Fast volume metric computation"""
    volumes = defaultdict(lambda: {"inflow": 0.0, "outflow": 0.0})
    
    # Single pass volume computation
    for tx in transactions:
        volumes[tx['from_addr']]["outflow"] += tx['token_amount']
        volumes[tx['to_addr']]["inflow"] += tx['token_amount']
    
    # Compute metrics only for addresses with significant volume
    total_volume = sum(v["inflow"] for v in volumes.values())
    volume_threshold = total_volume * 0.01  # 1% of total volume
    
    metrics = {
        'symmetry_scores': {},
        'concentration_ratios': {}
    }
    
    # Only analyze addresses with significant volume
    for addr, vol in volumes.items():
        if vol["inflow"] + vol["outflow"] > volume_threshold:
            min_flow = min(vol["inflow"], vol["outflow"])
            max_flow = max(vol["inflow"], vol["outflow"])
            metrics['symmetry_scores'][addr] = safe_division(min_flow, max_flow)
            metrics['concentration_ratios'][addr] = safe_division(
                vol["inflow"] + vol["outflow"], 
                2 * total_volume if total_volume else 1
            )
    
    return metrics

def find_approximate_cycles(G: nx.DiGraph, max_cycles: int = 10, max_length: int = 5) -> List[List[str]]:
    """Fast approximate cycle detection with limits"""
    cycles = []
    try:
        # Use a limited DFS for cycle detection
        for start_node in list(G.nodes())[:50]:  # Limit starting points
            if len(cycles) >= max_cycles:
                break
                
            stack = [(start_node, [start_node])]
            while stack and len(cycles) < max_cycles:
                current, path = stack.pop()
                
                # Look at neighbors
                for neighbor in G[current]:
                    if len(path) > max_length:
                        continue
                    if neighbor == start_node and len(path) > 2:
                        cycles.append(path)
                        break
                    if neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
    except:
        pass
    
    return cycles[:max_cycles]

def compute_cycle_metrics(transactions: List[Dict]) -> Dict:
    """Fast approximate cycle metric computation"""
    # Build graph with volume threshold
    G = nx.DiGraph()
    volume_threshold = np.percentile([tx['token_amount'] for tx in transactions], 25)
    
    for tx in transactions:
        if tx['token_amount'] >= volume_threshold:
            if G.has_edge(tx['from_addr'], tx['to_addr']):
                G[tx['from_addr']][tx['to_addr']]['weight'] += tx['token_amount']
            else:
                G.add_edge(tx['from_addr'], tx['to_addr'], weight=tx['token_amount'])
    
    # Find approximate cycles
    cycles = find_approximate_cycles(G)
    cycle_volumes = []
    
    for cycle in cycles:
        try:
            # Approximate cycle volume using average instead of min
            volumes = [G[cycle[i]][cycle[(i + 1) % len(cycle)]]['weight'] 
                      for i in range(len(cycle))]
            cycle_volumes.append(np.mean(volumes))
        except:
            continue
    
    return {
        'cycles': cycles,
        'num_cycles': len(cycles),
        'avg_cycle_length': safe_mean([len(c) for c in cycles]),
        'cycle_volumes': cycle_volumes
    }

def compute_delta_metrics(transactions: List[Dict]) -> Dict:
    """Fast delta metric computation"""
    delta_metrics = defaultdict(lambda: {'position_reversions': 0})
    
    # Use single window size and sample transactions
    window_size = 86400  # 1 day
    sample_size = min(1000, len(transactions))
    
    if sample_size < len(transactions):
        sampled_transactions = np.random.choice(
            transactions, 
            sample_size, 
            replace=False
        ).tolist()
    else:
        sampled_transactions = transactions
    
    # Compute initial balances
    balances = defaultdict(float)
    for tx in sampled_transactions:
        balances[tx['from_addr']] -= tx['token_amount']
        balances[tx['to_addr']] += tx['token_amount']
    
    # Only analyze addresses with significant balance
    significant_addresses = {
        addr for addr, bal in balances.items() 
        if abs(bal) > np.percentile(list(abs(x) for x in balances.values()), 25)
    }
    
    timestamps = sorted([tx['timestamp'] for tx in sampled_transactions])
    if not timestamps:
        return delta_metrics
    
    # Analyze position reversions
    for tx_time in timestamps[::10]:  # Sample every 10th timestamp
        window_start = tx_time - window_size
        window_txs = [tx for tx in sampled_transactions 
                     if window_start <= tx['timestamp'] <= tx_time]
        
        for addr in significant_addresses:
            net_position = sum(tx['token_amount'] if tx['to_addr'] == addr 
                             else -tx['token_amount'] if tx['from_addr'] == addr 
                             else 0 
                             for tx in window_txs)
            
            if abs(balances[addr]) > 0 and abs(net_position) < 0.01 * abs(balances[addr]):
                delta_metrics[addr]['position_reversions'] += 1
    
    return delta_metrics

def format_augmented_features(volume_metrics: Dict, cycle_metrics: Dict, delta_metrics: Dict) -> str:
    """Format metrics into readable string"""
    try:
        high_symmetry_addrs = [addr for addr, score in volume_metrics['symmetry_scores'].items() 
                              if score > 0.8]
        high_cycle_addrs = set([addr for cycle in cycle_metrics['cycles'] for addr in cycle])
        frequent_reverting_addrs = [addr for addr, metrics in delta_metrics.items() 
                                   if metrics['position_reversions'] > 5]

        concentration_values = list(volume_metrics['concentration_ratios'].values())
        min_concentration = min(concentration_values) if concentration_values else 0
        max_concentration = max(concentration_values) if concentration_values else 0

        return f"""
AUGMENTED ANALYSIS FEATURES:

1. Volume Patterns:
- High Volume Symmetry Addresses (>80% in/out balance):
  {', '.join(high_symmetry_addrs[:5]) if high_symmetry_addrs else 'None detected'}
- Volume Concentration Range: {min_concentration:.3f} to {max_concentration:.3f}

2. Circular Trading:
- Number of Detected Cycles: {cycle_metrics['num_cycles']}
- Average Cycle Length: {cycle_metrics['avg_cycle_length']:.2f}
- Addresses in Cycles: {len(high_cycle_addrs)}
- Average Cycle Volume: {safe_mean(cycle_metrics['cycle_volumes']):.2f}

3. Position Analysis:
- Addresses with Frequent Position Reversions: {len(frequent_reverting_addrs)}
- Average Position Reversion Frequency: {safe_mean([m['position_reversions'] for m in delta_metrics.values()]):.2f}

Key Suspicious Patterns:
- {len(set(high_symmetry_addrs) & high_cycle_addrs)} addresses show both high volume symmetry and cycle participation
- {len(set(frequent_reverting_addrs) & high_cycle_addrs)} addresses show both frequent position reversions and cycle participation
"""
    except Exception as e:
        print(f"Error formatting metrics: {e}")
        return "Error computing augmented features."

def compute_all_metrics(transactions: List[Dict]) -> str:
    """Compute and format all augmented metrics"""
    try:
        volume_metrics = compute_volume_metrics(transactions)
        cycle_metrics = compute_cycle_metrics(transactions)
        delta_metrics = compute_delta_metrics(transactions)
        
        return format_augmented_features(volume_metrics, cycle_metrics, delta_metrics)
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return "Error computing augmented features."