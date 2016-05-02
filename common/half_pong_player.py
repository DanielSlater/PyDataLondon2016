from collections import deque

from pygame.constants import K_DOWN
from pygame.constants import K_UP

from resources.PyGamePlayer.games.half_pong import run
from resources.PyGamePlayer.pygame_player import PyGamePlayer


class HalfPongPlayer(PyGamePlayer):
    SCREEN_WIDTH = 80
    SCREEN_HEIGHT = 80
    CUMULATIVE_SCORE_LEN = 1000
    ACTIONS_COUNT = 3

    def __init__(self, **kwargs):
        """
        Plays Half Pong by choosing moves randomly
        """
        super(HalfPongPlayer, self).__init__(**kwargs)
        self._last_score = 0
        self._score_history = deque()

    def get_feedback(self):
        from resources.PyGamePlayer.games.half_pong import score

        # get the difference in scores between this and the last frame
        score_change = score - self._last_score
        self._last_score = score
        self._score_history.append(score_change)

        if len(self._score_history) > self.CUMULATIVE_SCORE_LEN:
            self._score_history.popleft()

        return float(score_change), score_change == -1

    def cumulative_score(self):
        return sum(self._score_history)

    @staticmethod
    def action_index_to_key(action_index):
        if action_index == 0:
            return [K_DOWN]
        elif action_index == 1:
            return []
        else:
            return [K_UP]

    def start(self):
        super(HalfPongPlayer, self).start()

        run(screen_width=self.SCREEN_WIDTH, screen_height=self.SCREEN_HEIGHT)


if __name__ == '__main__':
    player = HalfPongPlayer()
    player.start()
