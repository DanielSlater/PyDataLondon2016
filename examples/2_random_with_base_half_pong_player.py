import random

from pygame.constants import K_DOWN, K_UP

from common.half_pong_player import HalfPongPlayer


class RandomHalfPongPlayer(HalfPongPlayer):
    def get_keys_pressed(self, screen_array, feedback, terminal):
        print self.score()

        action_index = random.randrange(3)

        return HalfPongPlayer.action_index_to_key(action_index)


if __name__ == '__main__':
    player = RandomHalfPongPlayer()
    player.start()