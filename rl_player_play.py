# -*- coding: utf-8 -*-
import asyncio
import rl_player_train as rlp
import numpy as np
import tensorflow as tf
import os

from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
from poke_env.player.player import Player

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session, get_session


class RLOutplayer(Player):
    model = tf.keras.models.load_model("model_7000")

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type and battle.opponent_active_pokemon:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def choose_move(self,battle):
        inputs = embed_battle(battle)
        ouputs = self.model.predict(inputs)


    if __name__ == "__main__":
        m = tf.keras.models.load_model("model_7000")
        m.summary()
        print(m.layers[0].input_shape)
        inp = np.concatenate([
        np.array([100,40,50,10]),
        np.array([1,1,1,1]),
        [5,4],])
        print(inp.shape)
        inp = inp.reshape(1,1,10)
        print(m.predict(inp))
