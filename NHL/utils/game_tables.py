# standard stuff
import pandas as pd

# formatting stuff
import unidecode


class FormatGames:

    def __init__(self, player_filepath=None):
        """
        Initialize the class
        :param player_filepath: csv containing the player names and their ids
        """

        # Initialize player DF
        # The csv listed here should include player names and player ids at the very least.
        self.player_filepath = player_filepath
        self.player_stats = pd.read_csv(player_filepath)

        # Get first and last names for comparison later
        self.player_stats['last_name'] = self.player_stats.player.apply(lambda x: x.split('.')[-1])
        self.player_stats['first_name'] = self.player_stats.player.apply(lambda x: x.split('.')[0])

        # Initalize game paths
        self.game_filepath = None
        self.game_stats = None

        self.last_mismatch = []
        self.first_mismatch = []

    def load_games_csv(self, game_filepath=None, years=None, **kwargs):
        """
        Load the game statistics csv into a df
        :param game_filepath: path to the game stats csv
        :param years: additional keyword if you want to load multiple csvs corresponding to different years of play
        :param kwargs: extra keywords for the read_csv function
        :return:
        """
        self.game_filepath = game_filepath
        if years is None:
            self.game_stats = pd.read_csv(game_filepath, **kwargs)
        else:
            year = years[0]
            year_path = self.game_filepath[:-4] + '_' + str(year) + '.csv'
            self.game_stats = pd.read_csv(year_path, **kwargs)
            for year in years[1:]:
                year_path = self.game_filepath[:-4]+'_'+str(year)+'.csv'
                year_stats = pd.read_csv(year_path, **kwargs)
                self.game_stats = self.game_stats.append(year_stats, ignore_index=True)

    def save_games_csv(self, new_filepath=None):
        """
        Save to a new file
        :param new_filepath: new file to save to
        :return: nothing
        """
        self.game_stats.to_csv(new_filepath, index=False)

    def format_games(self):
        """
        Format the games DF so it can be compared with the players DF
        :return: nothing
        """
        # Reformat to match the "FIRST.LAST" formatting with no spaces, no accents.
        self.game_stats.player = self.game_stats.player.apply(lambda x: x.replace(' ', '.'))
        self.game_stats.player = self.game_stats.player.apply(lambda x: unidecode.unidecode(x).upper())

        # Get their last names for comparison.
        self.game_stats['last_name'] = self.game_stats.player.apply(lambda x: x.split('.')[-1])

        # Also get first names.
        self.game_stats['first_name'] = self.game_stats.player.apply(lambda x: x.split('.')[0])

        self.first_mismatch, self.last_mismatch = self.match_names()

    def match_names(self):
        # assign the same player id to those that match the cap friendly DB
        # Need to match from the player information DF to the
        # game stats DF as the player info one is the one with unique ids already.
        """
        Assign the same player id to those that match the cap friendly DB
        Need to match from the player information DF to the
        game stats DF as the player info one is the one with unique ids already.
        :return: list of players not matched by first or last name
                 Will also print off some stats on how many where missed.
        """
        # Set their player id to -1
        self.game_stats['player_id'] = -1

        unique_players = self.player_stats.player_id.nunique()
        unique_players_from_games = self.game_stats.player.nunique()
        count_last_no_match = 0
        count_first_no_match = 0
        not_in_last = []
        not_in_first = []
        for i, last in enumerate(self.player_stats.last_name.values):
            # Get the ID of the player in question
            player_id = self.player_stats.loc[i, 'player_id']

            # If the last names don't even match, skip over this player.
            if last not in self.game_stats.last_name.values:
                count_last_no_match += 1
                not_in_last.append(last)
            else:
                # If the last names match, check to see if the game stats DF is using a nickname
                # (As the player info DF always uses full name)
                first = self.player_stats.first_name.values[i]
                first_name_list = self.game_stats[self.game_stats['last_name'] == last].first_name.values
                found = False

                # Could have more than one name sharing same last name, so check all of them.
                for first_from_games in list(set(first_name_list)):
                    if first_from_games in first:
                        found = True

                        # Sweet, we found a match!
                        # Set the player_id to match the correct name.
                        self.game_stats.loc[(self.game_stats.last_name == last) & (
                                    self.game_stats.first_name == first_from_games), 'player_id'] = player_id

                if not found:
                    count_first_no_match += 1
                    not_in_first.append((first, first_name_list, last))

        # Fix some outliers of people with the same name that I am aware of.
        names = ['NICK.JENSEN', 'NICKLAS.JENSEN']
        for name in names:
            player_id = self.player_stats.loc[self.player_stats.player == name, 'player_id'].values[0]
            self.game_stats.loc[self.game_stats.player == name, 'player_id'] = player_id

        print()
        print("Number of unique players in PLAYER DF:", unique_players)
        print("Original number of unique players in GAMES DF: ", unique_players_from_games)
        print("Number of unique players in GAMES DF that match PLAYER DF:", self.game_stats.player_id.nunique())
        print("Total number of unmatched last names: ", count_last_no_match)
        print("Total number of unmatched first names:", count_first_no_match)
        print()

        return not_in_first, not_in_last
