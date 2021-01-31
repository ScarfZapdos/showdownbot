# -*- coding: utf-8 -*-
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.environment.weather import Weather as w
from poke_env.environment.pokemon_type import PokemonType as t
import team
import asyncio
import numpy as np
import FirstPlayer
import random

#my_player_config = PlayerConfiguration("my_username", None)

class ExpertPlayer(Player):
    def choose_move(self, battle):


        if battle.active_pokemon.fainted:
            print("I lost a Pokemon :'(")
            if battle.weather == w.SANDSTORM:
                if not battle.team['p1: Dracozolt'].fainted:
                    print("Dracozolt i choose you")
                    return self.create_order(battle.team['p1: Dracozolt'])


        if battle.active_pokemon.type_1 == t.ELECTRIC: #Dracozolt
            if battle.weather == w.SANDSTORM:
                if battle.available_switches:
                    return self.create_order(random.choice(battle.available_switches))
            maxi = 0
            movetoret = None
            multiplier = None
            for m in battle.available_moves:
                multiplier = m.type.damage_multiplier(battle.opponent_active_pokemon.type_1,battle.opponent_active_pokemon.type_2)
                if multiplier > maxi:
                    maxi = multiplier
                    movetoret = m
            if movetoret != None:
                return self.create_order(movetoret)


        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

    def teampreview(self, battle):
        mon_performance = {}
        # For each of our pokemons
        for i, mon in enumerate(battle.team.values()):
            # We store their average performance against the opponent team
            mon_performance[i] = np.mean([
                self.teampreview_performance(mon, opp)
                for opp in battle.opponent_team.values()
            ])
        ordered_mons = sorted(mon_performance, key = lambda k: -mon_performance[k])
        # We start with the one we consider best overall
        return "/team " + ''.join([str(i + 1) for i in ordered_mons])

    def teampreview_performance(self,mon_a, mon_b):
        # We evaluate the performance on mon_a against mon_b as its type advantage
        a_on_b = b_on_a = -np.inf
        for type_ in mon_a.types:
            if type_:
                a_on_b = max(a_on_b, type_.damage_multiplier(*mon_b.types))
        for type_ in mon_b.types:
            if type_:
                b_on_a = max(b_on_a, type_.damage_multiplier(*mon_a.types))
        # Our performance metric is the different between the two
        return a_on_b - b_on_a


async def main():
    first_player = FirstPlayer.FirstPlayer(
        battle_format="gen8ou",
        team=team.team_1,
        max_concurrent_battles=10,
    )
    max_damage_player = ExpertPlayer(
        battle_format="gen8ou",
        team=team.team_1,
        max_concurrent_battles=10,
    )

# Now, let's evaluate our player
    await max_damage_player.battle_against(first_player, n_battles = 100)

    print(
        "Max damage player won %d / 100 battles"
        % max_damage_player.n_won_battles
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
