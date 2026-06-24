class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        current_x = init
        for _ in range(iterations):
            current_x -= learning_rate * 2 * current_x
        return round(current_x, 5)