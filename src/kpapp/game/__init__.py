from .base import Game
from .dance.game import DanceGame
from .demo import CatchGame
from .flappy.game import FlappyGame
from .fruitninja.game import FruitNinjaGame
from .golf.game import GolfGame
from .holeinwall.game import HoleInWallGame
from .template.game import TouchTargetsGame

REGISTRY: dict[str, type[Game]] = {
    "catch": CatchGame,
    "holeinwall": HoleInWallGame,
    "fruitninja": FruitNinjaGame,
    "flappy": FlappyGame,
    "dance": DanceGame,
    "golf": GolfGame,
    "template": TouchTargetsGame,  # reference for new games; off the menu
}
