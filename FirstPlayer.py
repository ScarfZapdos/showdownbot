# -*- coding: utf-8 -*-
from poke_env.player.player import Player

class FirstPlayer(Player):
    def choose_move(self, battle):
        return self.choose_random_move(battle)
