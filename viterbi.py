import pandas as pd


def tweet_train_parser(filename):
    data = open(filename, "r", encoding="utf8")
    datalines = data.readlines()
    tweets = []
    temp = []
    for item in datalines[1:]:
        if item != "\n":
            temp.append(item)
        else:
            tweets.append(temp)
            temp = []
    newtweets = []

    tweets = list(map(lambda x: map(lambda y: y[:-1].split("\t"), x), tweets))
    tweets = list(map(lambda x: list(x), tweets))

    newtweets = []
    
    for tweet in tweets:
        if tweet[0][0] == "RT" or "@USER" in tweet[0][0]:
            newtweets.append(tweet)
        else:
            newtweets[-1].extend("")
            newtweets[-1].extend(tweet)
    return tweets

def tweet_test_parser(filename):
    data = open(filename, "r", encoding="utf8")
    datalines = data.readlines()
    tweets = []
    temp = []
    for item in datalines:
        if item != "\n":
            temp.append(item)
        else:
            tweets.append(temp)
            temp = []
    newtweets = []
    
    tweets = list(map(lambda x: list(map(lambda y: y[:-1], x)), tweets))
    #tweets = list(map(lambda x: list(x), tweets))
    newtweets = []
    
    for tweet in tweets:
        if tweet[0] == "RT" or "@USER" in tweet[0]:
            newtweets.append(tweet)
        else:
            newtweets[-1].extend("")
            newtweets[-1].extend(tweet)
    return newtweets
    

def generate_trans_prob(filename):
    tweets = tweet_train_parser(filename)

    tags = open("twitter_tags.txt", "r", encoding="utf8")
    taglist = ["*"]
    
    tempdict = {}
    
    for tag in tags.readlines():
        taglist.append(tag[0])

    for tag in taglist:
        tempdict[tag] = 0
        
    trans_prob_dict = {}

    tags = open("twitter_tags.txt", "r", encoding="utf8")
    for tag in taglist:
        trans_prob_dict[tag] = tempdict.copy()

    for tweet in tweets:
        prev = "*"
        for word in tweet:
            if not word:
                continue
            now = word[1]
            trans_prob_dict[prev][now] += 1
            prev = now

        trans_prob_dict[prev]["*"] += 1
        prev = now

    for prevTag in trans_prob_dict:
        summa = sum(trans_prob_dict[prevTag].values())
        for aftTag in trans_prob_dict[prevTag]:
            trans_prob_dict[prevTag][aftTag] /= summa 
        
    df = pd.DataFrame.from_dict(trans_prob_dict, orient='index')
    df.to_csv("trans_probs.txt", sep=' ', mode='a')
    return df

def generate_output_prob(filename):
    tweets = tweet_train_parser(filename)

    tags = open("twitter_tags.txt", "r", encoding="utf8")
    taglist = ["*"]
    
    for tag in tags.readlines():
        taglist.append(tag[0])

    tagTemplate = {}        
    for tag in taglist:
        tagTemplate[tag] = 0

    database = {}
    
    for tweet in tweets:
        for word in tweet:
            if not word:
                continue
            
            string = word[0]
            tag = word[1]
            if string in database:
                database[string][tag] += 1
            else:
                database[string] = tagTemplate.copy()
                database[string][tag] = 1

    for word in database:
        summa = sum(database[word].values())
        for tag in database[word]:
            database[word][tag] /= summa

    df = pd.DataFrame.from_dict(database, orient='index')
    df.to_csv("output_probs.txt", sep=' ', mode='a')
    return df


class Trellis():
    def __init__(self, tweet, states, trans_probs, output_probs):
        self.terms = []
        self.table = []
        cleantags = states.copy()
        cleantags.remove("*")
        
        for word in tweet:
            #if not word in output_probs.index.values:
            #    print(word)
            self.terms.append(word)
            nodus = []
            
            for state in cleantags:
                nodus.append(Node(word, state))
            self.table.append(nodus)
            
        self.states = states
        self.trans_probs = trans_probs
        self.output_probs = output_probs
        self.output_word_list = list(self.output_probs.index.values)

    def trans(self, prevState, nextState):
        return self.trans_probs.loc[prevState,nextState]

    def output(self, word, state):
        if word in self.output_word_list:
            return self.output_probs.loc[word, state]
        else:
            return 1

    def Viterbi(self):
        for currNode in self.table[0]:
            currNode.prob = self.trans("*", currNode.state) * self.output(currNode.word, currNode.state)
        
        for rowIndex in range(1, len(self.table)):
            row = self.table[rowIndex]
            for currNode in row:
                maxNode = max(self.table[rowIndex - 1], key = lambda x: x.prob * self.trans(x.state, currNode.state) * self.output(currNode.word, currNode.state))
                currNode.setPrevious(maxNode)
                currNode.prob *= self.trans(maxNode.state, currNode.state) * self.output(currNode.word, currNode.state)
        
        
        maxNode = max(self.table[-1], key = lambda x: x.prob * (self.trans(x.state, "*")))

        bestTags = []
        while maxNode != None:
            bestTags.append(maxNode.state)
            maxNode = maxNode.prev

        return bestTags[::-1]
        
        

class Node():
    def __init__(self, word, state):
        self.word = word
        self.state = state
        self.prev = None
        self.prob = 1

    def setPrevious(self, prevNode):
        self.prev = prevNode

    def prob(self):
        return self.prob

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    tweets = tweet_test_parser(in_test_filename)
    trans_probs = pd.read_table(in_trans_probs_filename, delim_whitespace=True, index_col = 0)
    output_probs = pd.read_table(in_output_probs_filename, delim_whitespace=True, index_col = 0)
    
    tags = open(in_tags_filename, "r", encoding="utf8")
    taglist = ["*"]
    
    for tag in tags.readlines():
        taglist.append(tag[0])
    
    tweets = list(map(lambda x: Trellis(x, taglist, trans_probs, output_probs), tweets))
    viterbi = list(map(lambda x: x.Viterbi(), tweets))
    
    output_file  = open(out_predictions_filename, 'w')

    for tags in viterbi:
        for tag in tags:
            output_file.write(tag + "\n")
        output_file.write("\n");

    output_file.close()

#viterbi_predict("twitter_tags.txt","trans_probs.txt","output_probs.txt","twitter_dev_no_tag.txt","output_viterbi.txt")
