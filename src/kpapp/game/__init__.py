from .base import Game
from .demo import CatchGame
from .flappy.game import FlappyGame
from .fruitninja.game import FruitNinjaGame
from .holeinwall.game import HoleInWallGame
from .template.game import TouchTargetsGame

REGISTRY: dict[str, type[Game]] = {
    "catch": CatchGame,
    "holeinwall": HoleInWallGame,
    "fruitninja": FruitNinjaGame,
    "flappy": FlappyGame,
    "template": TouchTargetsGame,  # reference for new games; off the menu
}
