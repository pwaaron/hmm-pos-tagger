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

def generate_output_prob2(filename):
    tweets = tweet_train_parser(filename)

    tags = open("twitter_tags.txt", "r", encoding="utf8")
    taglist = ["*"]
    
    for tag in tags.readlines():
        taglist.append(tag[0])

    tagTemplate = {}        
    for tag in taglist:
        tagTemplate[tag] = 0

    database = {}
    database["-s"] = tagTemplate.copy()
    database["-ed"] = tagTemplate.copy()
    database["-ing"] = tagTemplate.copy()
    database["-ion"] = tagTemplate.copy()
    database["-al"] = tagTemplate.copy()
    database["-ive"] = tagTemplate.copy()
    database["-NONE"] = tagTemplate.copy()
    database["CAPSLOCK*"] = tagTemplate.copy()
    database["small*"] = tagTemplate.copy()

    
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

            if string.endswith('s'):
                database["-s"][tag] += 1
            elif string.endswith('ed'):
                database["-ed"][tag] += 1
            elif string.endswith('ing'):
                database["-ing"][tag] += 1
            elif string.endswith('ion'):
                database["-ion"][tag] += 1
            elif string.endswith('al'):
                database["-al"][tag] += 1
            elif string.endswith('ive'):
                database["-ive"][tag] += 1
            else:
                database["-NONE"][tag] += 1

            if string[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                database["CAPSLOCK*"][tag] += 1
            else:
                database["small*"][tag] += 1
                

    for word in database:
        summa = sum(database[word].values())
        for tag in database[word]:
            database[word][tag] /= summa

    df = pd.DataFrame.from_dict(database, orient='index')
    df.to_csv("output_probs2.txt", sep=' ', mode='a')
    return df


class Trellis2():
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
        self.output_word_list_lower = list(map(lambda x: x.lower(), self.output_word_list)) 

    def trans(self, prevState, nextState):
        return self.trans_probs.loc[prevState,nextState].item()

    def output(self, word, state):
        if word in self.output_word_list:
            return self.output_probs.loc[word, state].item()
        
        elif word.lower() in self.output_word_list_lower:
            word = self.output_word_list[self.output_word_list_lower.index(word.lower())]
            return self.output_probs.loc[word, state].item()
        
        elif word in ["CAPSLOCK*", "small*", "-s", "-ed", "-ing", "-ion", "-al", "-ive", "-NONE"]:
            return 1
        else:
            return self.unknownWord(word, state)

    def unknownWord(self, word, state):
        if isHashtag(word):
            if state == "#":
                return 1
            else:
                return 0
        elif isURL(word):
            if state == "U":
                return 1
            else:
                return 0
        elif isUSER(word):
            if state == "@":
                return 1
            else:
                return 0
        else:
            
            if word[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                probCaps = self.output("CAPSLOCK*", state)
            else:
                probCaps = self.output("small*", state)

            if word.endswith('s'):
                probSuffix = self.output("-s", state)
            elif word.endswith('ed'):
                probSuffix = self.output("-ed", state)
            elif word.endswith('ing'):
                probSuffix = self.output("-ing", state)
            elif word.endswith('ion'):
                probSuffix = self.output("-ion", state)
            elif word.endswith('al'):
                probSuffix = self.output("-al", state)
            elif word.endswith('ive'):
                probSuffix = self.output("-ive", state)
            else:
                probSuffix = self.output("-NONE", state)

            return probCaps * probSuffix
        

    def Viterbi(self):
        for currNode in self.table[0]:
            currNode.prob = self.trans("*", currNode.state) * self.output(currNode.word, currNode.state)
        
        for rowIndex in range(1, len(self.table)):
            row = self.table[rowIndex]
            for currNode in row:
                maxNode = max(self.table[rowIndex - 1],
                              key = lambda x: x.prob
                              * self.trans(x.state, currNode.state)
                              * self.output(currNode.word, currNode.state))
                currNode.setPrevious(maxNode)
                currNode.prob *= self.trans(maxNode.state, currNode.state) * self.output(currNode.word, currNode.state)
        
        
        maxNode = max(self.table[-1], key = lambda x: x.prob * (self.trans(x.state, "*")))

        bestTags = []
        while maxNode != None:
            bestTags.append(maxNode.state)
            maxNode = maxNode.prev

        return bestTags[::-1]
        
def isHashtag(word):
    return word[0] == "#"

def isURL(word):
    return word.startswith("http://") or word.startswith("https://")

def isUSER(word):
    return word.startswith("@USER")

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

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    tweets = tweet_test_parser(in_test_filename)
    trans_probs = pd.read_table(in_trans_probs_filename, delim_whitespace=True, index_col = 0)
    output_probs = pd.read_table(in_output_probs_filename, delim_whitespace=True, index_col = 0)
    
    tags = open(in_tags_filename, "r", encoding="utf8")
    taglist = ["*"]
    
    for tag in tags.readlines():
        taglist.append(tag[0])
    
    tweets = list(map(lambda x: Trellis2(x, taglist, trans_probs, output_probs), tweets))
    viterbi = list(map(lambda x: x.Viterbi(), tweets))
    
    output_file  = open(out_predictions_filename, 'w')

    for tags in viterbi:
        for tag in tags:
            output_file.write(tag + "\n")
        output_file.write("\n");

    output_file.close()
    
    
#generate_output_prob2("twitter_train.txt")
#viterbi_predict2("twitter_tags.txt","trans_probs.txt","output_probs2.txt","twitter_dev_no_tag.txt","output_viterbi2.txt")
