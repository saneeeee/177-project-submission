from washdetector.generator.models import GeneratorParams, WashCamouflageParams, WashCamouflageStrategy
from washdetector.generator.generator import TradingDataGenerator
from washdetector.utils.validation import validate_transactions
from washdetector.visualization.graph import create_transaction_graph
from washdetector.agent.wash_detector import analyze_wash_trading


def main(seed, camouflage: bool = False):
    if camouflage:
        params = GeneratorParams(
            seed=42,
            num_transactions=200,
            num_normal_traders=20,
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
        # starter test params
        # params = GeneratorParams(
        #     seed=seed,
        #     num_transactions=200,
        #     num_normal_traders=10,
        #     num_wash_groups=2,
        #     wash_group_sizes=(3, 4),
        #     wash_amounts=(10.0, 20.0),
        #     wash_tx_counts=(20, 30),
        #     time_span_days=30,
        # )
        
        # no camouflage, washers are identified
        params = GeneratorParams(
            seed=seed,
            num_transactions=300,
            num_normal_traders=100,
            num_wash_groups=1,
            wash_group_sizes=(4,),
            wash_amounts=(20.0,),
            wash_tx_counts=(30,),
            time_span_days=30,
        )

        # no camouflage, yes market makers, washers are identified
        params = GeneratorParams(
            seed=seed,
            num_transactions=300,
            num_normal_traders=100,
            num_wash_groups=1,
            wash_group_sizes=(4,),
            wash_amounts=(20.0,),
            wash_tx_counts=(30,),
            time_span_days=30,
            num_market_makers=5,
            market_maker_activeness=0.6,
        )

        params = GeneratorParams(
            seed=seed,
            num_transactions=300,
            num_normal_traders=100,
            num_wash_groups=1,
            wash_group_sizes=(4,),
            wash_amounts=(20.0,),
            wash_tx_counts=(30,),
            time_span_days=30,
            camouflage=WashCamouflageParams(
                strategy=WashCamouflageStrategy.HYBRID,
                normal_trade_probability=0.3,
                volume_leakage=0.2,
                normal_connections_per_washer=5,
                wash_amount_variance=0.3,
                min_time_between_wash=1.0
            )
        )

        params = GeneratorParams(
            seed=seed,
            num_transactions=1000,
            num_normal_traders=100,
            num_wash_groups=2,
            wash_group_sizes=(4, 7),
            wash_amounts=(20.0, 10.0),
            wash_tx_counts=(30, 20),
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

    generator = TradingDataGenerator(params)
    transactions = generator.generate_transactions()
    is_valid = validate_transactions(transactions)
    
    print(f"Generated {len(transactions)} transactions")
    print(f"Valid sequence: {is_valid}")
    

    # print(transactions)
    analysis = analyze_wash_trading(transactions, OPENAI_API_KEY)

    #visualize 
    print("Creating transaction graph visualization...")
    graph_path = create_transaction_graph(transactions)
    print(f"Graph saved to: {graph_path}")
    
    return transactions, analysis

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # seeds = [42, 123, 456]

    # for seed in seeds:
    #     print(f"\nRunning evaluation with seed: {seed}")
    #     transactions, analysis = main(seed, camouflage=True)
    #     print(analysis)
    
    transactions, analysis = main(1, False)
    print(analysis)

