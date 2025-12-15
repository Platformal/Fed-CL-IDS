"""Extension of Opacus' Privacy Engine to get epsilon per round."""
import warnings
from opacus import PrivacyEngine

warnings.filterwarnings('ignore')

class DifferentialPrivacy(PrivacyEngine):
    def __init__(
            self, *,
            accountant: str = 'rdp',
            secure_mode: bool = False) -> None:
        super().__init__(accountant=accountant, secure_mode=secure_mode)
        self.previous_epsilon: float = 0.0

    def get_epsilon(self, delta: float) -> float:
        """Return how much epsilon was used at delta."""
        return super().get_epsilon(delta)
        total_epsilon = super().get_epsilon(delta)
        epsilon_used = total_epsilon - self.previous_epsilon
        self.previous_epsilon = total_epsilon
        return epsilon_used
