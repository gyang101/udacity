#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:04:35 2017

@author: gyang100
"""

"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def distance_from_center(game, player):
    """
    Output a score base on the distance between the center and player location
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).
    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    
    # Get the location of the center
    # taken from center_score
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    
    # distance formula sqrt((Y1 - Y2)**2 + (X1 - X2)**2)
    distance_from_center = math.sqrt((h - y)**2 + (w - x)**2)
    
    return float(distance_from_center)

def custom_score(game, player):
    """
    Checks
    Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return distance_from_center(game, player)


def run_away(game, player):
    """
    Calculate the distance between you and the opponent. Function attempts to
    keep a distance from the opponent
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    # Get players location
    own_loc = game.get_player_location(player)
    
    # Get opponents location
    opp_loc = game.get_player_location(game.get_opponent(player))

    return float(abs(sum(opp_loc) - sum(own_loc)))


def custom_score_2(game, player):
    """
    Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return run_away(game, player)


def square_legal_moves(game, player):
    """
    Returns value of own_moves - opp_moves to the second power.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float((own_moves - opp_moves)**2)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return square_legal_moves(game, player)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)
        
        legal_moves = game.get_legal_moves(game.active_player)
        # If no best move is found within the search time. Initialize best_move
        # to first available move to allow the game to be played out without forfeting
        if len(legal_moves) != 0:
            best_move = legal_moves[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
       Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Stores list of legal moves
        legal_moves = game.get_legal_moves(game.active_player)
        
        best_score = float('-inf')
        best_move = None
        
        # If no best move is found within the search time. Initialize best_move
        # to first available move to allow the game to be played out without forfeting
        if len(legal_moves) != 0:
            best_move = legal_moves[0]
        
        # For loop used to compare current_score with best_score and current_move with best_move.
        # Replaces best_score with current_score if best_score > current_score
        # Replaces best_move with current_move if best_move > current_move
        for current_move in legal_moves:
            current_score = self.min_value(game.forecast_move(current_move), depth - 1)
            if current_score > best_score:
                best_score = current_score
                best_move = current_move
                
        return best_move
                
    def terminal_test(self, state, depth):
        """
        Function used to determine if the game is over, if return is true
        """
        # Timer used to stop search for time efficiency
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Checks to see if depth is 0. If it is the recursion is stop as it would be inefficient time comsumption.    
        if depth == 0:
            return True
        
        # Stores legal moves into a list using functions from isolation.py
        legal_moves = state.get_legal_moves(state.active_player)
        
        # Checks to see if there are any legal moves left. Return true if legal_moves contains no legal moves
        # and returns false if there are still legal moves within legal_moves
        if len(legal_moves) == 0:
            return True
        else:
            return False
        
            
    def max_value(self, state, depth):
        """
        Function is used to determine the best maximum value and returns it
        """
         # Timer used to stop search for time efficiency
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Calls terminal test to determine if game is over
        if self.terminal_test(state, depth):
            # Returns a current state
            return self.score(state, state.active_player)
        
        # Predefine best score as negative infinity
        best_score = float("-inf")  
        # Stores legal moves into a list
        legal_moves = state.get_legal_moves(state.active_player)
        
        # For loop used to find the best maximum value move base on the opposing players
        # best mimumum value moves. Mutually recursive loop.
        for current_move in legal_moves:
            current_score = self.min_value(state.forecast_move(current_move), depth-1)
            # Compares if current_score > best_score. Then stores it as the best_score
            if current_score > best_score:
                best_score = current_score
        return best_score
    
    def min_value(self, state, depth):
        """
        Function is used to determine the best minumum value and returns it
        """
         # Timer used to stop search for time efficiency
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
         # Calls terminal test to determine if game is over
        if self.terminal_test(state, depth):
            # Returns a current state
            return self.score(state, state.inactive_player)
        
        # Predefine best score as infinity
        best_score = float("inf")
        # Stores legal moves into a list
        legal_moves = state.get_legal_moves(state.active_player)
        
        # For loop used to find the minumum value move base on the opposing players
        # maximum value moves. Mutually recursive loop.
        for current_move in legal_moves:
            current_score = self.max_value(state.forecast_move(current_move), depth-1)
            # Compares if current_score < best_score. Then stores it as the best_score
            if current_score < best_score:
                best_score = current_score
        return best_score   


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        best_move = (-1, -1)
        
        # Stores legal moves into a list
        legal_moves = game.get_legal_moves(game.active_player)
        
        # If no best move is found within the search time. Initialize best_move
        # to first available move to allow the game to be played out without forfeting
        if len(legal_moves) != 0:
            best_move = legal_moves[0]
            
        # The try/except block will automatically catch the exception
        # raised when the timer is about to expire.
        # Iterative deepening
        try:
            #Continue to apply the search while increasing the depth until time runs out
            for deepening_depth in range(0, 9999):
                best_move = self.alphabeta(game, deepening_depth)
                
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Stores list of legal moves
        legal_moves = game.get_legal_moves(game.active_player)
        
        best_score = float('-inf')
        best_move = (-1, -1)
        
        # If no best move is found within the search time. Initialize best_move
        # to allow the game to be played out without forfeting
        if len(legal_moves) != 0:
            best_move = legal_moves[0]
            
        # For loop used to compare current_score with best_score and current_move with best_move.
        # Replaces best_score with current_score if best_score > current_score
        # Replaces best_move with current_move if best_move > current_move
        for current_move in legal_moves:
            current_score = self.min_value(game.forecast_move(current_move), depth - 1, alpha, beta)
            if current_score > best_score:
                best_score = current_score
                best_move = current_move
            
            # Update alpha for pruning
            alpha = max(alpha, best_score)
                        
        return best_move
        
    def terminal_test(self, state, depth):
        """
        Function used to determine if the game is over, if return is true
        """
        # Timer used to stop search for time efficiency
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Checks to see if depth is 0. If it is the recursion is stop as it would be inefficient time comsumption.    
        if depth == 0:
            return True
        
        # Stores legal moves into a list using functions from isolation.py
        legal_moves = state.get_legal_moves(state.active_player)
        
        # Checks to see if there are any legal moves left. Return true if legal_moves contains no legal moves
        # and returns false if there are still legal moves within legal_moves
        if len(legal_moves) == 0:
            return True
        else:
            return False
        
    def max_value(self, gameState, depth, alpha, beta):
        """ 
        This function uses alpha-beta pruning to find the maximum value max can 
        get or else it returns -1 to indicate game has ended
        """
        # timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Terminal test
        if self.terminal_test(gameState, depth):
            return self.score(gameState, gameState.active_player)
        legal_moves = gameState.get_legal_moves(gameState.active_player)
        best_score = float("-inf")  
        for move in legal_moves:
            best_score = max(best_score, self.min_value(gameState.forecast_move(move), depth-1, alpha, beta))
            # Pruning. best_score is compare to beta to see if it is greater then or equal to beta
            if best_score >= beta:
                return best_score
            # Finds the largest value between alpha and best_score
            alpha = max(alpha, best_score)
        return best_score    
            
    def min_value(self, gameState, depth, alpha, beta):
        """ 
        This function uses alpha-beta pruning to find the minimum value min can
        get or else it returns -1 to indicate game has ended.
        """
        # timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Terminal Test
        if self.terminal_test(gameState, depth):
            return self.score(gameState, gameState.inactive_player)
        legal_moves = gameState.get_legal_moves(gameState.active_player)
        best_score = float("inf")
        for move in legal_moves:
            best_score = min(best_score, self.max_value(gameState.forecast_move(move), depth-1, alpha, beta))
            # Pruning. Best_score is compare to alpha to see if it is less then  or equal alpha.
            if best_score <= alpha:
                return best_score
            # Finds the minimum value between beta and best_score
            beta = min(beta, best_score)
        return best_score
    
    
