import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': log_likevalue, 'SOMEOTHERWORD' log_likevalue, ... },
            {SOMEWORD': log_likevalue, 'SOMEOTHERWORD' log_likevalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    for X, lengths in test_set.get_all_Xlengths().values():
        # Define the log of the likelihood
        log_like = {}                          
        # Define best_guess
        best_guess = None   
        # Define best_score
        best_score = float("-inf")                
        
        for word, model in models.items():
            try:
                # Define log if the likelihood
                current_score = model.score(X, lengths) 
                log_like[word] = current_score             
                
                # If current_score is higher than previous best_score
                # replace best_score with current current_score
                if current_score > best_score:
                    best_score = current_score
                    best_guess = word
            except:
                # If model is not valid, throw out
                log_like[word] = float("-inf")
        
        # Append best word to guesses list
        guesses.append(best_guess)
        # Append log of the likelihood to the list
        probabilities.append(log_like)

    return probabilities, guesses
