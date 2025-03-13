import pytest
import copy

class TestIndexExtremeStrategy:
    """Test for the IndexExtremeStrategy methods without requiring a backtrader environment."""
    
    @pytest.fixture
    def strategy_params(self):
        """Return a dictionary of strategy parameters."""
        return {
            "low_percentile": 20,
            "high_percentile": 80,
            "extreme_low_percentile": 10,
            "extreme_high_percentile": 90,
            "expert_mode": False,
            "expert_signal": None
        }
    
    def adjust_position(self, params, spread_pctl, positions, x_idx, y_idx, expert_signal=None):
        """
        Extracted version of the adjust_position method from IndexExtremeStrategy.
        
        Args:
            params (dict): Strategy parameters
            spread_pctl (float): Percentile of the spread
            positions (list): Current positions in [7yr, 5yr, 3yr]
            x_idx (int): Position index of longer-term bond
            y_idx (int): Position index of shorter-term bond
            expert_signal (int, optional): Expert signal, 0 for rates down, 1 for rates up
            
        Returns:
            list (float): Adjusted positions in [7yr, 5yr, 3yr]
        """
        # Check expert_signal is valid
        if params["expert_mode"]:
            if expert_signal is None:
                raise ValueError("Expert signal is required when expert mode is enabled")
            elif expert_signal not in [0, 1]:
                raise ValueError("Expert signal must be 0 or 1")

        # Use the parameters for percentile thresholds
        low_pctl = params["low_percentile"] / 100
        high_pctl = params["high_percentile"] / 100
        extreme_low = params["extreme_low_percentile"] / 100
        extreme_high = params["extreme_high_percentile"] / 100

        # Make a copy of positions to avoid modifying the original
        positions = positions.copy()

        # If using expert mode, use expert signal to adjust positions
        if params["expert_mode"]:
            if spread_pctl <= low_pctl and expert_signal == 1:
                # Low spread + Rates up expectation, shift to extremely shorter position.
                positions[y_idx] += 2
                positions[x_idx] -= 2
            elif spread_pctl <= low_pctl and expert_signal == 0:
                # Low spread + Rates down expectation, shift to shorter position.
                positions[y_idx] += 1
                positions[x_idx] -= 1
            elif spread_pctl >= high_pctl and expert_signal == 0:
                # High spread + Rates down expectation, shift to extremely longer position.
                positions[x_idx] += 2
                positions[y_idx] -= 2
            elif spread_pctl >= high_pctl and expert_signal == 1:
                # High spread + Rates up expectation, shift to longer position.
                positions[x_idx] += 1
                positions[y_idx] -= 1
        # If not using expert mode, use four-tier percentile thresholds
        else:
            if spread_pctl <= extreme_low:
                positions[y_idx] += 2
                positions[x_idx] -= 2
            elif extreme_low < spread_pctl <= low_pctl:
                positions[y_idx] += 1
                positions[x_idx] -= 1
            elif high_pctl <= spread_pctl < extreme_high:
                positions[x_idx] += 1
                positions[y_idx] -= 1
            elif extreme_high <= spread_pctl:
                positions[x_idx] += 2
                positions[y_idx] -= 2

        # Ensure all positions are non-negative
        positions = [max(0, p) for p in positions]

        # If total position exceeds target_sum(normally 6 units), scale proportionally
        target_sum = 6
        total = sum(positions)
        if total > target_sum:
            positions = [p * target_sum / total for p in positions]
            
        # Round positions to 1 decimal place
        positions = [round(p, 1) for p in positions]
        
        # Ensure total equals exactly target_sum (6.0) with a single adjustment
        total = sum(positions)
        if total != target_sum:
            diff = target_sum - total
            if diff > 0:
                # Find the smallest position and add the remaining amount
                min_idx = positions.index(min(positions))
                positions[min_idx] = round(positions[min_idx] + diff, 1)
            else:
                # Find the largest position and subtract the excess
                max_idx = positions.index(max(positions))
                positions[max_idx] = round(positions[max_idx] + diff, 1)  # diff is negative
                
        return positions

    def get_final_positions(self, params, spread_53, spread_57, spread_73):
        """
        Extracted version of the get_final_positions method from IndexExtremeStrategy.
        
        Args:
            params (dict): Strategy parameters
            spread_53 (float): 5yr-3yr spread percentile
            spread_57 (float): 7yr-5yr spread percentile
            spread_73 (float): 7yr-3yr spread percentile
            
        Returns:
            list: Final positions [7yr, 5yr, 3yr]
        """
        # Initial positions: [7yr, 5yr, 3yr] 2 units each
        positions = [2, 2, 2]

        # Apply adjustments in sequence
        # For 5yr-3yr spread: x_idx=1 (5yr), y_idx=2 (3yr)
        positions = self.adjust_position(
            params, spread_53, positions, 1, 2, params["expert_signal"]
        )

        # For 7yr-5yr spread: x_idx=0 (7yr), y_idx=1 (5yr)
        positions = self.adjust_position(
            params, spread_57, positions, 0, 1, params["expert_signal"]
        )

        # For 7yr-3yr spread: x_idx=0 (7yr), y_idx=2 (3yr)
        positions = self.adjust_position(
            params, spread_73, positions, 0, 2, params["expert_signal"]
        )

        return positions

    def test_adjust_position_equals_target_sum(self, strategy_params):
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
            params = copy.deepcopy(strategy_params)
            params["expert_mode"] = expert_mode
            params["expert_signal"] = 1 if expert_mode else None
            
            for spread_pctl, positions, x_idx, y_idx in test_cases:
                # Call adjust_position method
                result = self.adjust_position(
                    params, spread_pctl, positions, x_idx, y_idx, 
                    expert_signal=1 if expert_mode else None
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
                    # Check if decimal part is a multiple of 0.1
                    decimal_part = round(p * 10) % 10 / 10
                    assert abs(p - (int(p) + decimal_part)) < 1e-10, (
                        f"Position {p} not rounded to 1 decimal place"
                    )

    def test_get_final_positions_equals_target_sum(self, strategy_params):
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
            params = copy.deepcopy(strategy_params)
            params["expert_mode"] = expert_mode
            params["expert_signal"] = 1 if expert_mode else None
            
            for spread_53, spread_57, spread_73 in test_cases:
                # Get final positions
                result = self.get_final_positions(
                    params, spread_53, spread_57, spread_73
                )
                
                # Check that positions sum to exactly 6.0
                assert sum(result) == pytest.approx(6.0), (
                    f"Positions {result} don't sum to 6.0 with spreads: "
                    f"spread_53={spread_53}, spread_57={spread_57}, spread_73={spread_73}"
                ) 