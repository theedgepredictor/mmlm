import re

from simulation.team import Team


class Game:
    '''
    Game class is an object for each tournament slot that is
    populated as the tournament continues. It also holds functions
    relavent to a game like updating teams from the results dict
    and returning a winner based on the predictions from the
    submission class.
    '''

    def __init__(self, row_slots, t_dict, s_dict, season):
        # Add relavent metadata for game - source is slots csv
        self.season = season
        self.slot = row_slots['Slot']
        self.strong_seed = row_slots['StrongSeed']
        self.weak_seed = row_slots['WeakSeed']

        # extract round label from game
        r = re.compile(r'(R.)[WXYZC].')
        match = r.search(self.slot)
        if match is not None:
            self.r_label = match.group(1)
        else:
            self.r_label = 'R0'  # label play-in games

        # set round equiv to tournament.current_r (int)
        self.r = int(self.r_label[-1])

        # Set teams if slot is determined only by seed
        #       This places only the initial games.
        self.strong_team = None
        self.weak_team = None
        strong_id = s_dict.get(self.strong_seed)
        weak_id = s_dict.get(self.weak_seed)

        # Initiate team class that holds team attrib.
        if strong_id is not None:
            self.strong_team = Team(strong_id,
                                    t_dict.get(strong_id),
                                    self.strong_seed)

        if weak_id is not None:
            self.weak_team = Team(weak_id,
                                  t_dict.get(weak_id),
                                  self.weak_seed)

    def __repr__(self):
        if self.team_is_missing():
            return f'{self.season} - {self.slot}: Game not yet set'
        else:
            return (f'{self.season} - {self.slot}: {self.strong_team.name} '
                    f'vs. {self.weak_team.name}')

    @property
    def game_id(self):
        return '_'.join([str(self.season),
                         str(self.strong_team.id),
                         str(self.weak_team.id)])

    def add_teams(self, results):
        '''
        Checks all results and updates games if results exist.
        '''

        if results.get(self.strong_seed) is not None:
            self.strong_team = results.get(self.strong_seed)
        if results.get(self.weak_seed) is not None:
            self.weak_team = results.get(self.weak_seed)

    def team_is_missing(self):
        '''
        Checks if either team is missing
        '''
        if self.strong_team is None or self.weak_team is None:
            return True
        else:
            return False

    def get_winner(self, submission, style, seed=0):
        '''
        Retrieves the winner of the game from the submission
        file based on the chosen methodology.
        '''

        if self.team_is_missing():
            raise ValueError('At least one team does not exist')

        if style == 'chalk':
            win_id = (
                submission.get_pred(self.game_id)
                          .get_favored()
                          )
            return win_id
        elif style == 'random':
            win_id = (
                submission.get_pred(self.game_id)
                          .get_random()
                          )
            return win_id
        else:
            raise ValueError('Please choose style=random or chalk')
