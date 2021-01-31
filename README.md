# Showdown Bot
By Hugo MARTIN

## poke-env API

Lors de ce projet , j'utilise l'API de [H. Sahovic](https://github.com/hsahovic) : [poke-env](https://github.com/hsahovic/poke-env)
Cette API permet de communiquer avec le serveur de [Pokemon Showdown](https://play.pokemonshowdown.com/) et de gérer toutes les informations échangées à travers des classes python.  
Cette API est particulièrement pensée pour le RL.

## Implementation

Dans ce dépôt, on retrouve principalement les exemples fournis dans la documentation officielle de l'API, et un implémentation de joueur à base de DQN dans les fichiers **rl_player_***. Il utilise keras-rl2.
Ce joueur bat le joueur aléatoire avec 99% de chances.