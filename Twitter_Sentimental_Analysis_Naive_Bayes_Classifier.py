
#################################################################################

#######################Twitter Sentimental Analysis##############################

##################################################################################


import re
import csv
import pprint
import nltk.classify
from itertools import groupby


def HashTagSplit(text):
    
    text = re.sub(r'#([^\s]+)', r'\1', text)
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    j = 0
    while 0 < i:
        words.append(text[lasts[i]:i])
        j = j+1
        i = lasts[i]
    words.reverse()
    finalword = ""


    for i in range(0,j):
        finalword = finalword +'\t' +words[i]

    return finalword #probs[-1]

def word_prob(word): return dictionary.get(word, 0) / total
def words(text): return re.findall('[a-z]+', text.lower()) 

dictionary = dict((w, len(list(ws)))
                  for w, ws in groupby(sorted(words(open('data/words.txt').read()))))
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))


############################################################################
#Replace SMS lingo words


def smslingo(message):

    alphaindex = {}
    count = []

    vocab = {}
    abb = []

    dictionary = [[0 for x in range(200)] for x in range(800)]




    f1 = open('data/dictionary1.txt',"r+")
    f2 = open('data/dictionary2.txt',"r+")
    f3 = open('data/count.txt',"r+")
    f5 = open('data/label.txt',"r+")


    tmp = 0
    put = 0

    for words in f1.readlines():
        vocab[tmp]=words.lower()
        tmp = tmp +1

    



    tmp=0

    for words in f2.readlines():
        words=words.lower()
        abb.append(words)
        tmp = tmp +1

    counter= 0


    for words in f3.readlines():
        count.append(words)
        counter = counter+1

    alpha =0

    for words in f5.readlines():
        alphaindex[words]=alpha
        alpha = alpha + 1


    track = 0
    put =0
    get = 0
    

    for i in range(0,counter,27):
        for j in range(1,28):
            temporary = count[track]
            temporary = int(temporary)
            for i in range(0,temporary):
                first = vocab[get].lower().rstrip('\n')
                second = abb[get].lower().rstrip('\n')
                get = get + 1
                dictionary[put][i] = {first:second}
            put = put+1
            track  = track  +1

    keep = 0
    #messages = smstext.split()
    words = message

   
    words = words.lower()
    first = words[0]
    if(len(words)>=2):
        second = words[1]
        letter = first + second
    else:
        letter = first
    
    for k,v in alphaindex.items():
        if set (letter.split()) & set(k.split()):
            index = v
            keep = 1
    if keep==0:
        letter = first
        for k,v in alphaindex.items():
            if set (letter.split()) & set(k.split()):
                index = v
    
    
    loop  = count[index]
    lt = int(loop)
    for i in range(0,lt):
        for k,v in dictionary[index][i].items():
            if set (words.split()) & set(k.split()):
                keep = 2
                replace = v

    

  

    if keep==2:
        replaceword = replace
        
    else:
        replaceword = words
    return replaceword
#end



#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    #tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace hash tag words after spliting them
    tweet = re.sub(r'#([^\s]+)', lambda x: HashTagSplit(x.group()), tweet)
	#trim
    tweet = tweet.strip('\'"')
    return tweet
#end 

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            w = smslingo(w)
            featureVector.append(w.lower())
    return featureVector    
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end

    



def naivebayesclassifier(train,test):
    print "Preparing the Training Data........."
    import time
    time.sleep(5)
    
    #Read the tweets one by one and process it
    inpTweets = csv.reader(open(train, 'rt'), delimiter=',', quotechar='|')
    stopWords = getStopWordList('data/stopwords.txt')
    count = 0;
    featureList = []
    tweets = []

    inputtweet = {}
    outputtweet = {}

    precision = []
    recall = []
    f1 = []
    tp =[]
    fn = []
    fp = []
    for i in range(0,3):
        precision.append(0.000000)
        recall.append(0.000000)
        f1.append(0.000000)
        tp.append(0.000000)
        fn.append(0.000000)
        fp.append(0.000000)

    for row in inpTweets:
        sentiment = row[0]
        tweet = row[1]
        labels.append(sentiment)
        print tweet
        print '\n'
        processedTweet = processTweet(tweet)
        featureVector = getFeatureVector(processedTweet, stopWords)
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment));
    #end loop

    # Remove featureList duplicates
    featureList = list(set(featureList))

    f6 = open('featurelist.txt','w+')

    #Create Features vector list

    for features in featureList:
        newfeatures = features + '\t'
        f6.write(newfeatures)
        

    # Generate the training set
    training_set = nltk.classify.util.apply_features(extract_features, tweets)

    # Train the Naive Bayes classifier
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

    # Test the classifier
    print "Classifying the Tweets........."
    import time
    time.sleep(5)


    inpTweets = csv.reader(open(test, 'rt'), delimiter=',', quotechar='|')


    f2=open('output.txt', 'w+')

    for row in inpTweets:
        inputsentiment = row[0]
        line = row[1]
        print line
        print '\n'
        processedTestTweet = processTweet(line)
        sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
        line = line.rstrip('\n')
        f2.write(line+'\t'+sentiment)
        f2.write('\n')
        inputtweet[line] = inputsentiment
      
        outputtweet[line] = sentiment
        



    #calculate F1 score

    for i in range(0,3):

        if(i==0):
            check = 'positive'
        elif(i==1):
            check = 'negative'
        elif(i==3):
            check = 'neutral'

        inpTweets = csv.reader(open(test, 'rt'), delimiter=',', quotechar='|')
        for row in inpTweets:
            line = row[1]
            line = line.rstrip('\n')
            for k,v in inputtweet.items():
                if(line==k):
                    inputvalue = v
            for k,v in outputtweet.items():
                if(line==k):
                    outputvalue = v
            if((inputvalue == check) & (outputvalue ==check)):
                tp[i] = tp[i] +1
            else:
                if(inputvalue == check):
                    fp[i] = fp[i] +1
                elif(outputvalue == check):
                    fn[i] = fn[i] +1
        
        precision[i] = float((tp[i]) / (tp[i] + fp[i]))
        a = tp[i] + fn[i]
        b =tp[i]
        recall[i] = float(b/a)
        f1[i] = float((2 * precision[i] * recall[i] ) / (precision[i] + recall[i]))

    #macro F1

    macrof1 = ((f1[0] + f1[1] + f1[2] ) / 3) * 100

    print "Macro F1 Score for Naive Bayes Classifer is : " + macrof1


#Main

labels = []

print "##################################################"
print '\n'
print "           Twitter Sentimental Analysis"
print '\n'
print "##################################################"
print '\n'
print '\n'
print "Enter the Train file path"

train = raw_input()

print "Enter the Test file path"

test = raw_input()

#naive bayes Classification
naivebayesclassifier(train,test)



    
    
        

        


        

    












