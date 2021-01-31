# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session, get_session

#DEEPLY INSPIRED BY POKE-ENV EXAMPLE

tf_config = tf.compat.v1.ConfigProto(
device_count={'GPU': 0},
intra_op_parallelism_threads=1,
allow_soft_placement=True)

tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6

tf.keras.backend.clear_session()
sess = tf.compat.v1.Session(config=tf_config)
graph = tf.compat.v1.get_default_graph()



# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
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

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


NB_TRAINING_STEPS = 5000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)


# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps):
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        dqn.fit(player, nb_steps=nb_steps)
        player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            player.reset_battles()
            dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

            print(
            "DQN Evaluation: %d victories out of %d episodes"
            % (player.n_won_battles, nb_episodes)
            )


if __name__ == "__main__":
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")

    print(env_player.action_space)

    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    # Output dimension

    n_action = len(env_player.action_space)

    model = Sequential()
    model.add(Dense(128, activation="elu", name='Input', input_shape=(1, 10)))

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten(name='Flatten1'))
    model.add(Dense(64, name='Hidden1', activation="elu"))
    model.add(Dense(n_action, name='Hidden2', activation="linear"))

    memory = SequentialMemory(limit=5000, window_length=1)

    # Simple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=5000,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=750,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )


    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    dqn.compile(Adam(lr=0.00025), metrics=["mae"])

    # Training
    with graph.as_default():
        env_player.play_against(
            env_algorithm=dqn_training,
            opponent=opponent,
            env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS},
    )
        model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
