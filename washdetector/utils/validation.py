from typing import List, Dict

def validate_transactions(transactions: List[dict], initial_balance: float = 1000.0) -> bool:
    """Validate transaction sequence for balance consistency"""
    # Get unique addresses
    addresses = set()
    for tx in transactions:
        addresses.add(tx['from_addr'])
        addresses.add(tx['to_addr'])
    
    # Initialize balances
    balances = {addr: initial_balance for addr in addresses}
    
    for tx in transactions:
        amount = tx['token_amount']
        
        if balances[tx['from_addr']] < amount:
            return False
            
        balances[tx['from_addr']] -= amount
        balances[tx['to_addr']] += amount
        
    return True
