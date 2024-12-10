import random
from datetime import datetime
import uuid
from typing import List, Dict, Set
import numpy as np
from collections import defaultdict

from .models import GeneratorParams, WashCamouflageStrategy

class TradingDataGenerator:
    def __init__(self, params: GeneratorParams):
        self.params = params
        random.seed(params.seed)
        np.random.seed(params.seed)
        
        self.addresses = self._generate_addresses()
        self.market_makers = self._setup_market_makers()
        self.wash_groups = self._setup_wash_groups()
        self.balances = {addr: 1000.0 for addr in self.addresses}
        self.start_time = int(datetime(2024, 1, 1).timestamp())
        
        # Initialize metrics tracking
        self.metrics = {
            'wash_transactions': [],
            'normal_transactions': [],
            'market_maker_transactions': [],
            'camouflage_transactions': []
        }
        
        # track wash trader connections for camouflage
        self.wash_trader_connections = self._setup_wash_trader_connections()
        
    def _generate_deterministic_address(self, index: int) -> str:
        random.seed(self.params.seed + index)
        return 'r' + uuid.UUID(int=random.getrandbits(128)).hex[:39]

    def _generate_addresses(self) -> List[str]:
        addresses = []
        index = 0
        
        # generate addresses for normal traders and market makers
        for _ in range(self.params.num_normal_traders):
            addr = self._generate_deterministic_address(index)
            addresses.append(addr)
            index += 1
        
        # generate addresses for wash traders
        for size in self.params.wash_group_sizes:
            for _ in range(size):
                addr = self._generate_deterministic_address(index)
                addresses.append(addr)
                index += 1
                
        return addresses

    def _setup_market_makers(self) -> Set[str]:
        """Setup market maker addresses from normal traders"""
        market_makers = set()
        if self.params.num_market_makers > 0:
            market_makers = set(self.addresses[:self.params.num_market_makers])
            print("\nMarket Makers:")
            for addr in market_makers:
                print(f"  {addr}")
            print()
        return market_makers

    def _setup_wash_groups(self) -> List[List[str]]:
        """Create wash trading groups and print their addresses"""
        groups = []
        start_idx = self.params.num_normal_traders
        
        print("Wash Trading Groups:")
        for group_idx, size in enumerate(self.params.wash_group_sizes):
            group = self.addresses[start_idx:start_idx + size]
            groups.append(group)
            
            print(f"\nGroup {group_idx + 1} (Size: {size}):")
            for addr in group:
                print(f"  {addr}")
                
            start_idx += size
        print()
        return groups

    def _setup_wash_trader_connections(self) -> Dict[str, Set[str]]:
        """Setup camouflage connections for wash traders"""
        connections = defaultdict(set)
        if self.params.camouflage and self.params.camouflage.strategy in [
            WashCamouflageStrategy.RANDOM_CONNECT, 
            WashCamouflageStrategy.HYBRID
        ]:
            normal_traders = set(self.addresses[:self.params.num_normal_traders]) - self.market_makers
            for group in self.wash_groups:
                for washer in group:
                    num_connections = self.params.camouflage.normal_connections_per_washer
                    if num_connections > 0:
                        connections[washer] = set(random.sample(
                            list(normal_traders), 
                            min(num_connections, len(normal_traders))
                        ))
        return connections

    def _generate_camouflage_transactions(self, wash_trader: str, timestamp: int) -> List[Dict]:
        """Generate camouflage transactions for a wash trader"""
        transactions = []
        
        if (self.params.camouflage and 
            random.random() < self.params.camouflage.normal_trade_probability):
            connected_traders = self.wash_trader_connections[wash_trader]
            if connected_traders:
                trader = random.choice(list(connected_traders))
                if random.random() < 0.5:
                    from_addr, to_addr = wash_trader, trader
                else:
                    from_addr, to_addr = trader, wash_trader
                
                amount = max(0.1, np.random.normal(
                    self.params.normal_trade_size_mean,
                    self.params.normal_trade_size_std
                ))
                
                tx = {
                    'tx_id': str(uuid.UUID(int=random.getrandbits(128))),
                    'timestamp': timestamp,
                    'from_addr': from_addr,
                    'to_addr': to_addr,
                    'token_amount': amount,
                    'token_symbol': self.params.token_symbol,
                    'price_eth': self._generate_price(),
                    'tx_type': 'camouflage'
                }
                transactions.append(tx)
                self.metrics['camouflage_transactions'].append(tx)
        
        return transactions

    def _generate_price(self) -> float:
        """Generate price with volatility"""
        return 1.0 * (1 + random.uniform(-self.params.price_volatility, 
                                       self.params.price_volatility))

    def get_ground_truth_info(self) -> Dict:
        """Return ground truth information about the generated data"""
        total_volume = 0
        wash_volume = 0
        
        # Calculate volumes
        for tx in self.metrics['wash_transactions']:
            wash_volume += tx['token_amount']
        
        for metric_list in self.metrics.values():
            for tx in metric_list:
                total_volume += tx['token_amount']
        
        return {
            'wash_groups': self.wash_groups,
            'market_makers': list(self.market_makers),
            'wash_trader_connections': {
                k: list(v) for k, v in self.wash_trader_connections.items()
            },
            'metrics': {
                'total_volume': total_volume,
                'wash_volume': wash_volume,
                'wash_volume_percentage': wash_volume / total_volume if total_volume > 0 else 0,
                'num_wash_transactions': len(self.metrics['wash_transactions']),
                'total_transactions': sum(len(txs) for txs in self.metrics.values())
            }
        }

    def generate_transactions(self) -> List[dict]:
        transactions = []
        time_span = self.params.time_span_days * 24 * 3600
        
        random.seed(self.params.seed)
        np.random.seed(self.params.seed)

        # generate wash trading transactions
        for group_idx, group in enumerate(self.wash_groups):
            wash_amount = self.params.wash_amounts[group_idx]
            num_cycles = self.params.wash_tx_counts[group_idx]
            
            for cycle in range(num_cycles):
                base_timestamp = self.start_time + (cycle * time_span // num_cycles)
                
                # add temporal variance if using temporal camouflage
                if (self.params.camouflage and 
                    self.params.camouflage.strategy in [
                        WashCamouflageStrategy.TEMPORAL, 
                        WashCamouflageStrategy.HYBRID
                    ]):
                    base_timestamp += random.randint(
                        int(self.params.camouflage.min_time_between_wash * 3600),
                        int(time_span / num_cycles)
                    )
                
                group_order = list(group)
                random.Random(self.params.seed + group_idx + cycle).shuffle(group_order)
                
                # generate wash cycle
                for i in range(len(group_order)):
                    from_addr = group_order[i]
                    to_addr = group_order[(i + 1) % len(group_order)]
                    
                    # add amount variance if using volume-based camouflage
                    actual_amount = wash_amount
                    if (self.params.camouflage and 
                        self.params.camouflage.strategy in [
                            WashCamouflageStrategy.VOLUME_BASED, 
                            WashCamouflageStrategy.HYBRID
                        ]):
                        variance = wash_amount * self.params.camouflage.wash_amount_variance
                        actual_amount += random.uniform(-variance, variance)
                    
                    tx = {
                        'tx_id': str(uuid.UUID(int=random.getrandbits(128))),
                        'timestamp': base_timestamp + i * 3600,
                        'from_addr': from_addr,
                        'to_addr': to_addr,
                        'token_amount': actual_amount,
                        'token_symbol': self.params.token_symbol,
                        'price_eth': self._generate_price(),
                        'tx_type': 'wash'
                    }
                    transactions.append(tx)
                    self.metrics['wash_transactions'].append(tx)
                    
                    # generate camouflage transactions
                    transactions.extend(self._generate_camouflage_transactions(
                        from_addr, 
                        base_timestamp + i * 3600
                    ))

        # generate normal trading transactions
        remaining_tx = self.params.num_transactions - len(transactions)
        normal_traders = set(self.addresses[:self.params.num_normal_traders])
        non_market_makers = normal_traders - self.market_makers
        
        for i in range(remaining_tx):
            timestamp = self.start_time + (i * time_span // remaining_tx)
            
            # prefer market makers as counterparties based on activeness
            if self.market_makers and random.random() < self.params.market_maker_activeness:
                if random.random() < 0.5:
                    from_addr = random.choice(list(self.market_makers))
                    to_addr = random.choice(list(non_market_makers))
                else:
                    from_addr = random.choice(list(non_market_makers))
                    to_addr = random.choice(list(self.market_makers))
                tx_type = 'market_maker'
                metric_key = 'market_maker_transactions'
            else:
                from_addr = random.choice(list(normal_traders))
                to_addr = random.choice(list(normal_traders - {from_addr}))
                tx_type = 'normal'
                metric_key = 'normal_transactions'
            
            amount = max(0.1, np.random.normal(
                self.params.normal_trade_size_mean,
                self.params.normal_trade_size_std
            ))
            
            tx = {
                'tx_id': str(uuid.UUID(int=random.getrandbits(128))),
                'timestamp': timestamp,
                'from_addr': from_addr,
                'to_addr': to_addr,
                'token_amount': amount,
                'token_symbol': self.params.token_symbol,
                'price_eth': self._generate_price(),
                'tx_type': tx_type
            }
            transactions.append(tx)
            self.metrics[metric_key].append(tx)

        # Sort by timestamp before returning
        transactions.sort(key=lambda x: x['timestamp'])
        return transactions
