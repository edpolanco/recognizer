import warnings
from asl_data import SinglesData
import operator



def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
   
    #iterate all words in test data
    for word_id in range(len(test_set.wordlist)):
        X,lengths = test_set.get_item_Xlengths(word_id)
       
        #lets create our probability dictionary
        prob_word_dic = {}
        for word, model in models.items():
            try:
                prob_word_dic[word] = model.score(X,lengths)
            except:
                pass
           
        #update probabilities list
        probabilities.append(prob_word_dic)
        
        #get max value logL and then 
        # print('finding max item')
        guess_word = max(prob_word_dic.items(), key=operator.itemgetter(1))[0]
        guesses.append(guess_word)

    return (probabilities,guesses)
