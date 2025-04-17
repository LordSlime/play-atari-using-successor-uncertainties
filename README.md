# play-atari-using-successor-uncertainties
A simply program to visualize and save the result of using SU algorithm to play atari game

Before using this program to visualize and save the result of using SU algorithm to play atari game, you need to train successor_uncertainties_atari first through the original author's github page:
https://github.com/DavidJanz/successor_uncertainties_atari

After training, put the play_atari.py downloaded through this page under successor_uncertainties_atari folder.

Then run play_atari.py with command below:
python play_atari.py --game {the game name}
The game names are the same as atari games' name and make sure the state_dict you loaded matches the game.

