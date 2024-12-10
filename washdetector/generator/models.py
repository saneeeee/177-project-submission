from pydantic import BaseModel, Field, validator
from typing import Tuple, Optional
from enum import Enum

class WashCamouflageStrategy(str, Enum):
    NONE = "none"  # No camouflage
    RANDOM_CONNECT = "random_connect"  # Random connections to normal traders
    VOLUME_BASED = "volume_based"  # Mix wash volumes with normal trading volumes
    TEMPORAL = "temporal"  # Spread wash trades temporally among normal trades
    HYBRID = "hybrid"  # Combination of multiple strategies

class WashCamouflageParams(BaseModel):
    strategy: WashCamouflageStrategy = Field(default=WashCamouflageStrategy.NONE)
    # Probability of a wash trader engaging in normal trading
    normal_trade_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    # What fraction of wash trading volume goes to normal trades
    volume_leakage: float = Field(default=0.0, ge=0.0, le=1.0)
    # How many normal traders each wash trader connects to
    normal_connections_per_washer: int = Field(default=0, ge=0)
    # How much to vary wash amounts to look like normal trades
    wash_amount_variance: float = Field(default=0.0, ge=0.0, le=1.0)
    # Minimum time between wash trades (in hours)
    min_time_between_wash: float = Field(default=0.0, ge=0.0)
    
class GeneratorParams(BaseModel):
    seed: int = Field(..., description="Random seed for reproducibility")
    num_transactions: int = Field(..., ge=1)
    num_normal_traders: int = Field(..., ge=2)
    num_wash_groups: int = Field(..., ge=0)
    wash_group_sizes: Tuple[int, ...] = Field(default=())
    wash_amounts: Tuple[float, ...] = Field(default=())
    wash_tx_counts: Tuple[int, ...] = Field(default=())
    time_span_days: int = Field(..., ge=1)
    token_symbol: str = Field(default="TOKEN1")  # TODO 0.1: single token assumption
    price_volatility: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # New parameters for normal trading patterns
    normal_trade_size_mean: float = Field(default=50.0, ge=0.0)
    normal_trade_size_std: float = Field(default=20.0, ge=0.0)
    normal_trade_frequency_mean: float = Field(default=24.0, ge=0.0)  # hours
    normal_trade_frequency_std: float = Field(default=12.0, ge=0.0)   # hours
    
    # Camouflage parameters
    camouflage: WashCamouflageParams = Field(default_factory=WashCamouflageParams)
    
    # Market maker simulation
    num_market_makers: int = Field(default=0, ge=0)
    market_maker_activeness: float = Field(default=0.5, ge=0.0, le=1.0)

    @validator('wash_group_sizes', 'wash_amounts', 'wash_tx_counts')
    def validate_wash_params(cls, v, values):
        if 'num_wash_groups' in values:
            if len(v) != values['num_wash_groups']:
                raise ValueError(
                    f'Length of wash parameters must match num_wash_groups: {values["num_wash_groups"]}'
                )
        return v

    @validator('num_market_makers')
    def validate_market_makers(cls, v, values):
        if v > 0 and 'num_normal_traders' in values:
            if v >= values['num_normal_traders']:
                raise ValueError('Number of market makers must be less than number of normal traders')
        return v
