from washdetector.generator.models import GeneratorParams, WashCamouflageParams, WashCamouflageStrategy
from washdetector.generator.generator import TradingDataGenerator
from washdetector.utils.validation import validate_transactions
from washdetector.visualization.graph import create_transaction_graph

def main(camouflage: bool = False):
    if camouflage:
        params = GeneratorParams(
            seed=42,
            num_transactions=1000,
            num_normal_traders=50,
            num_wash_groups=2,
            wash_group_sizes=(3, 4),
            wash_amounts=(10.0, 20.0),
            wash_tx_counts=(20, 30),
            time_span_days=30,
            num_market_makers=5,
            market_maker_activeness=0.6,
            camouflage=WashCamouflageParams(
                strategy=WashCamouflageStrategy.HYBRID,
                normal_trade_probability=0.3,
                volume_leakage=0.2,
                normal_connections_per_washer=5,
                wash_amount_variance=0.3,
                min_time_between_wash=1.0
            )
        )
    else:
        params = GeneratorParams(
            seed=42,
            num_transactions=1000,
            num_normal_traders=50,
            num_wash_groups=2,
            wash_group_sizes=(3, 4),
            wash_amounts=(10.0, 20.0),
            wash_tx_counts=(20, 30),
            time_span_days=30,
        )

    generator = TradingDataGenerator(params)
    transactions = generator.generate_transactions()
    is_valid = validate_transactions(transactions)
    
    print(f"Generated {len(transactions)} transactions")
    print(f"Valid sequence: {is_valid}")
    
    #verify determinism
    gen2 = TradingDataGenerator(params)
    transactions2 = gen2.generate_transactions()
    assert transactions == transactions2, "Generator is not deterministic!"
    
    #visualize 
    print("Creating transaction graph visualization...")
    graph_path = create_transaction_graph(transactions)
    print(f"Graph saved to: {graph_path}")
    
    return transactions

if __name__ == "__main__":
    transactions = main(True)

