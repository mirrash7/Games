// The game registry (browser twin of game/__init__.py). Games with a title and
// `menu !== false` appear on the welcome screen, in this order.

import { SnackAttackGame } from "./snack/game.js";
import { FlappyRaccoonGame } from "./flappy/game.js";

export const GAMES = [SnackAttackGame, FlappyRaccoonGame];
