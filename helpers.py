from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn


''' function delete_stopwords(sentence) 
input: one sentence as string, output: this sentence withot stopwords as string
functional: delete standart inglish stopword from one sentence
using libraries: stopwords, word_tokenize
example:
    input: 'This is a sample sentence, showing off the stop words filtration.'
    output: 'This sample sentence , showing stop words filtration .'
    '''    
def delete_stopwords(sentence):
    sentence_token = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_token = [w for w in sentence_token if not w in stop_words]
    filtered_sentence = ' '.join(filtered_token)
    return filtered_sentence

''' function get_synonims(sentence_list) 
input: list of the sentences, output: list of the similar sentence
functional: for each sentence from input list delete stopwords, 
            for other words - looking for the closest by meaning synonim, if exist - replace original word by this synonim
using libraries: wordnet, word_tokenize
example:
    input: ['This is an example of sentence.', 'One more sentence in the list.']
    output: ['This model conviction.', 'One conviction tilt.']
    ''' 
def get_synonims(sentence_list):
    synonims_list = []    
    for quest in sentence_list:        
        filtered_sentence = delete_stopwords(quest)
        filtered_token = word_tokenize(filtered_sentence)
        text = ""
        for word in filtered_token:
            synonim = word
            counter = 1
            while synonim == word:
                try:
                    closest_synonim = wn.synset(word + ".n.0" + str(counter))
                    words = closest_synonim.name().split('.')
                    synonim = words[0].replace('_', ' ')
                    counter += 1
                except Exception:
                    break
            text += synonim + " "
        synonims_list.append(text)
    return synonims_list