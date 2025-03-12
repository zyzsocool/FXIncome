import pytest
import numpy as np
from fxincome.backtest.index_extreme_bactest import IndexExtremeStrategy


class TestIndexExtremeStrategy:
    @pytest.fixture
    def strategy(self):
        """Create and return a strategy instance for testing."""
        return IndexExtremeStrategy()

    def test_adjust_position_equals_target_sum(self, strategy):
        """Test that adjust_position always returns positions that sum to exactly 6.0."""
        # Define test cases: (spread_pctl, initial_positions, x_idx, y_idx)
        test_cases = [
            # Standard case with equal positions
            (0.05, [2, 2, 2], 0, 1),  # Low percentile
            (0.95, [2, 2, 2], 0, 1),  # High percentile
            
            # Unequal initial positions
            (0.05, [3, 1, 2], 0, 2),
            (0.95, [1, 2, 3], 1, 2),
            
            # Edge cases that could cause rounding issues
            (0.50, [1.9, 2.1, 2.0], 0, 1),
            (0.75, [2.33, 1.84, 1.83], 0, 2),
            
            # Case where positions could go negative before clamping
            (0.05, [0.5, 5.0, 0.5], 0, 1),
            (0.95, [0.5, 0.5, 5.0], 0, 2),
        ]
        
        # Test each case with both expert modes
        for expert_mode in [True, False]:
            strategy.p.expert_mode = expert_mode
            
            # Set percentile thresholds for testing
            strategy.p.low_percentile = 20
            strategy.p.high_percentile = 80
            strategy.p.extreme_low_percentile = 10
            strategy.p.extreme_high_percentile = 90
            
            for spread_pctl, positions, x_idx, y_idx in test_cases:
                # Test with expert signal if in expert mode
                expert_signal = 1 if expert_mode else None
                
                # Call adjust_position method
                result = strategy.adjust_position(
                    spread_pctl, positions.copy(), x_idx, y_idx, expert_signal
                )
                
                # Check that positions sum to exactly 6.0
                assert sum(result) == pytest.approx(6.0), (
                    f"Positions {result} don't sum to 6.0 with inputs: "
                    f"spread_pctl={spread_pctl}, positions={positions}, "
                    f"x_idx={x_idx}, y_idx={y_idx}, expert_mode={expert_mode}"
                )
                
                # Check all positions are non-negative
                assert all(p >= 0 for p in result), (
                    f"Negative position found in {result}"
                )
                
                # Check all positions are rounded to 1 decimal place
                for p in result:
                    decimal_part = abs(p - round(p, 0))
                    assert decimal_part == 0.0 or decimal_part == pytest.approx(0.1), (
                        f"Position {p} not rounded to 1 decimal place"
                    )

    def test_get_final_positions_equals_target_sum(self, strategy):
        """Test that get_final_positions always returns positions summing to 6.0."""
        # Test with various spread percentile combinations
        test_cases = [
            (0.05, 0.05, 0.05),  # All low
            (0.95, 0.95, 0.95),  # All high
            (0.05, 0.95, 0.50),  # Mixed
            (0.50, 0.50, 0.50),  # All middle
            (0.15, 0.85, 0.50),  # Mixed around thresholds
        ]
        
        # Test with both expert modes
        for expert_mode in [True, False]:
            strategy.p.expert_mode = expert_mode
            strategy.p.expert_signal = 1 if expert_mode else None
            
            # Set percentile thresholds
            strategy.p.low_percentile = 20
            strategy.p.high_percentile = 80
            strategy.p.extreme_low_percentile = 10
            strategy.p.extreme_high_percentile = 90
            
            for spread_53, spread_57, spread_73 in test_cases:
                # Get final positions
                result = strategy.get_final_positions(spread_53, spread_57, spread_73)
                
                # Check that positions sum to exactly 6.0
                assert sum(result) == pytest.approx(6.0), (
                    f"Positions {result} don't sum to 6.0 with spreads: "
                    f"spread_53={spread_53}, spread_57={spread_57}, spread_73={spread_73}"
                ) 