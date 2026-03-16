"""Base strategy class and strategy registry.

All strategies inherit from BaseStrategy and register via @StrategyEngine.register().
"""

from abc import ABC, abstractmethod
from models.types import Signal, Candle


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: dict):
        self.name: str = config["strategy"]["name"]
        self.config: dict = config
        self.params: dict = config.get("parameters", {})
        self.risk_config: dict = config.get("risk", {})
        self._active: bool = True

    @abstractmethod
    async def on_candle(self, token: str, candle: Candle) -> Signal | None:
        """Called when a new candle completes. Return a Signal or None."""

    @abstractmethod
    async def on_tick(self, token: str, price: float) -> Signal | None:
        """Called on each price tick. Return a Signal or None.
        Default: do nothing (override for tick-level strategies)."""

    @abstractmethod
    async def initialize(self):
        """Called once at startup. Load historical data, compute initial indicators."""

    async def teardown(self):
        """Called on shutdown. Clean up resources."""
        pass

    def is_active(self) -> bool:
        return self._active

    def deactivate(self):
        self._active = False

    def activate(self):
        self._active = True


class StrategyEngine:
    """Registry pattern for strategy discovery and instantiation."""

    REGISTRY: dict[str, type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy class.

        Usage:
            @StrategyEngine.register("rsi_bounce")
            class RSIBounceStrategy(BaseStrategy):
                ...
        """
        def wrapper(strategy_cls: type[BaseStrategy]):
            cls.REGISTRY[name] = strategy_cls
            return strategy_cls
        return wrapper

    @classmethod
    def create(cls, config: dict) -> BaseStrategy:
        """Instantiate a strategy from config."""
        strategy_type = config["strategy"]["type"]
        if strategy_type not in cls.REGISTRY:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available: {list(cls.REGISTRY.keys())}"
            )
        return cls.REGISTRY[strategy_type](config)

    @classmethod
    def load_all(cls, strategies_config: list[dict]) -> dict[str, BaseStrategy]:
        """Load all enabled strategies from config."""
        strategies = {}
        for sc in strategies_config:
            if not sc["strategy"].get("enabled", True):
                continue
            strategy = cls.create(sc)
            strategies[strategy.name] = strategy
        return strategies
