# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------

# Web Technology Coursework 
# Feb/March 2018 
# Christian Johnston (mdsw22)
# Multi-document summarisation
# Using python 3 

# To run: python3 TermFrequency.py summaryLength
# Where summaryLength is a single integer

#Libraries used
from difflib import SequenceMatcher
from operator import itemgetter
from nltk import tokenize
from rake_nltk import Rake
import numpy as np
import sys
import math

#------------------------------------------------------------------------------
#input files given as a list 
files = ["Doc 1.txt", "Doc 2.txt", "Doc 3.txt", "Doc 4.txt", "Doc 5.txt",
"Doc 6.txt", "Doc 7.txt", "Doc 8.txt"]

number_of_files = len(files)

summaryLength = int(sys.argv[1])

documentLengths = []

#List of keywords 
# ***** I know this makes it non-autonomous *****
# included for proof of concept 
keyWords = ["elon", "rocket", "spacex", "space", "musk", "tesla", "roadster", "heavy",
"falcon", "dragon", "tunnel"]

#lexical / phrasal summary cue
# cuewords often used before summarising a point containing crucial info.
cueWords = ["conclusion", "summary", "consequence", "result", "thus", "therefore", "thereby"]


# Rapid Automatic Keyword Extraction algorithm to generate keywords
# If word analysed is one of these keywords, multiply weight by scalar
r = Rake()
keyWords2 = []

#List of pronouns 
pronouns = ["who", "what", "why", "when", "how", "where", "me", "him", "her", "it", "them", "whom", 
"mine", "yours", "his", "hers", "ours", "theirs", "this", "that", "these", "those",
 "who", "whom", "which", "what", "whose", "whoever", "whatever", "whichever", "whomever",
 "himself", "herself", "itself", "themselves"]

#List of stop words
stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards']
stopwords += ['again', 'against', 'all', 'almost', 'alone', 'along']
stopwords += ['already', 'also', 'although', 'always', 'am', 'among']
stopwords += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
stopwords += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
stopwords += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
stopwords += ['because', 'become', 'becomes', 'becoming', 'been']
stopwords += ['before', 'beforehand', 'behind', 'being', 'below']
stopwords += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
stopwords += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
stopwords += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de']
stopwords += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
stopwords += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
stopwords += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
stopwords += ['every', 'everyone', 'everything', 'everywhere', 'except']
stopwords += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
stopwords += ['five', 'for', 'former', 'formerly', 'forty', 'found']
stopwords += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
stopwords += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
stopwords += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
stopwords += ['herself', 'him', 'himself', 'his', 'how', 'however']
stopwords += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
stopwords += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
stopwords += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
stopwords += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
stopwords += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
stopwords += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
stopwords += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
stopwords += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
stopwords += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
stopwords += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
stopwords += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
stopwords += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
stopwords += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
stopwords += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
stopwords += ['some', 'somehow', 'someone', 'something', 'sometime']
stopwords += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
stopwords += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
stopwords += ['then', 'thence', 'there', 'thereafter', 'thereby']
stopwords += ['therefore', 'therein', 'thereupon', 'these', 'they']
stopwords += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
stopwords += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
stopwords += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
stopwords += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
stopwords += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
stopwords += ['whatever', 'when', 'whence', 'whenever', 'where']
stopwords += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
stopwords += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
stopwords += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
stopwords += ['within', 'without', 'would', 'yet', 'you', 'your']
stopwords += ['yours', 'yourself', 'yourselves']

#------------------------------------------------------------------------------

#function to remove stop words
def removeStopwords(words, stopwords):
    return [w for w in words if w not in stopwords]

#fucntion to open file and read it in a string
def openAndReadFile(fileName):
    file = open(fileName, "r")
    text_string = file.read()
    return text_string

# function to format text string by removing all non ASCII symbols
def formatText(text_string):
    #convert to lower case 
    text_string = text_string.lower()
    for ch in '!"#$%&()*+-./,:;<=>?@':
        text_string = text_string.replace(ch, '')
    # remove numbers
    text_string = ''.join([i for i in text_string if not i.isdigit()])
    return text_string

#Function that breaks a string into its sentences
# split on full stop with a space after

def getSentences(text_string):
    sentences = tokenize.sent_tokenize(text_string)
    for sentence in sentences:
        sentence = formatText(sentence)
        sentence = sentence.replace('\n','')
    return sentences

# Using python Natural Language Toolkit to split up a file into sentences
def splitSentences(text_string):
    sentences = tokenize.sent_tokenize(text_string)
    return sentences

# Function to create term matrix
#each row is a word, each column a document, the entry being the number of times that word appears in that document
def getTermMatrix():

    #Creating term matrix
    termMatrix = []

    #need a counter for column indexing, i.e. document number (note starts from 1)
    columnCounter = 1;

    # loop through all the files
    for fileName in files:
        _file = openAndReadFile(fileName);
        #keyword extraction
        r.extract_keywords_from_text(_file)
        keyWords2.append(r.get_ranked_phrases())

        text = formatText(_file);
        # use space as a delimiter, i.e. get list of words
        words = text.split()
        #remove stop words
        words = removeStopwords(words, stopwords)

        #loop through all the words
        for word in words:
            found = False;
            #loop through all the rows
            for i in range (0,len(termMatrix)):
                #if found, incremenet count in correct column
                if word == termMatrix[i][0]:
                    termMatrix[i][columnCounter] += 1
                    #set found to true
                    found = True;

            if(not found):
                #create new row for that word
                #Set count to 0 for all previous documents
                newrow = [0] * (number_of_files+1)
                #Set count to 1 for current document
                newrow[columnCounter] = 1
                newrow[0] = word
                termMatrix.append(newrow)
        # increment column counter when moving onto new document
        columnCounter += 1;

    return termMatrix

#function to get the word weight for a given document
def getWordWeight(word, matrix, column):
    weight = 0;
    if word in stopwords:
        return 0.0
    #loop through all the rows
    for i in range (0,len(termMatrix)):
        if word == termMatrix[i][0]:
            weight = termMatrix[i][column]
    # if sentence contains a keyword, multiple the sentence weight by a scaling factor
    if word in keyWords:
        weight = weight *1.5
    if word in keyWords2:
        weight = weight*1.2
    return weight

# Calculate sentence weights- sum the word weights and normalise.
# for a given column / document
def getSentenceWeight(sentence, matrix, column):
    sentenceWeight = 0.0;
    wordWeight = 0.0;
    words = sentence.split()
    numWords = len(words)
    if numWords == 0:
        return 0
    for word in words:
        wordWeight = getTf_idf(word, matrix, column)
        sentenceWeight += wordWeight
        if word in cueWords:
            sentenceWeight = weight*1.2 

    # normalising so sentence weight not biased to length of sentence  
    averageSentenceWeight = sentenceWeight/numWords
    
    #normalising sentence weight so weight is not biased to sentences in longer documents
    # need the -1 as I have started column/document indexing from 1 (follows format of file names)
    averageSentenceWeight = averageSentenceWeight / documentLengths[column-1]
    return averageSentenceWeight

#this functions gets all the sentences from the documents
#returns them as a list of form: sentence, sentenceWeight, column( i.e which document is it in), lineNo (where in document sentence appears) 
def getSentenceList(matrix):
    sentenceDict = []
    column = 1;
    for fileName in files:
        _file = openAndReadFile(fileName)
        _file = _file.replace("\n", " ")

        sentences = splitSentences(_file)
        #get the length of each document 
        documentLengths.append(len(sentences))
        lineNo = 1
        for sentence in sentences:
            formattedSentence = formatText(sentence)
            sentenceWeight = getSentenceWeight(formattedSentence, matrix, column)
            sentenceDict.append([sentence, sentenceWeight, column, lineNo])
            lineNo += 1
        column += 1
    return sentenceDict

# Function to incoprorate location information into the sentence weights
# Sentence reference index, gives more weight to a sentence that 
# precedes sentence that containing a pronominal reference. 
# If a sentence contains a pronoun then the weight of the preceding sentence is increased.
def stuctureInfo(sentenceList):
    scaling = 1.2
    for i in range(1,len(sentenceList)):
        for pronoun in pronouns:
            if pronoun in sentenceList[i][0]:
                sentenceList[i-1][1] *= scaling
    return sentenceList

#Inverted pyramid 
#locational info- sentences at beginning of document contain the most important info
# thus given a higher weighting.
def locationInfo(sentenceList):
    for i in range(0,len(sentenceList)-1):
        if sentenceList[i][3] == 1:
            sentenceList[i][3] *= 2
        elif sentenceList[i][3] == 2:
            sentenceList[i][3] *= 1.5
        elif sentenceList[i][3] == 3:
            sentenceList[i][3] *= 1.3
        elif sentenceList[i][3] == 4:
            sentenceList[i][3] *= 1.2 
        elif sentenceList[i][3] == 5:
            sentenceList[i][3] *= 1.1 
        # final sentence also very important, often summarises document
        # also given a higher weighting 
        # if sentence (i+1) is the start of a new document
        # sentence i is the final sentence of previous document
        elif sentenceList[i+1][3] == 1:
            sentenceList[i][3] *= 1.3
    return sentenceList

def shortSummary(sentenceList):
    for i in range(0,len(sentenceList)):
        sentence = sentenceList[i][0]
        numWords = len(sentence.split())
        if(numWords > 15):
            sentenceList[i][1] *= 0.5
        if(numWords > 10):
            sentenceList[i][1] *= 0.8
    return sentenceList

# this loops through all the sentences and selects one of greatest weight
# also returns the index of found sentence
def getBestSentence(sentenceList, matrix):
    bestSentence = None;
    bestSentenceWeight = 0.0;
    index = 0;
    for i in range (0,len(sentenceList)):
        sentenceWeight = sentenceList[i][1]
        if sentenceWeight > bestSentenceWeight:
            bestSentenceWeight = sentenceWeight
            bestSentence = sentenceList[i][0]
            index = i;
    return bestSentence, index

# function to determine number of words in a sentence
def getNumberOfWords(sentence):
    words = sentence.split();
    numWords = len(words)
    return numWords

# rank how similar chosen sentences are, and ignore them if they are too similar.
# returns a value between 0 and 1
def similar(sentence1, sentence2):
    return SequenceMatcher(None, sentence1, sentence2).ratio()

#------------------------------------------------------------------------------
# function which acts as main function

def getSummary(requiredLength, sentenceList, matrix):
    length = 0;
    report = []

    #if(requiredLength < 300):

        #length of every best sentence chosen must be smaller???
        #box packing algorithm ?
        #use as many of the available words as possible

    while length < requiredLength:
        bestSentence, index = getBestSentence(sentenceList, matrix)
        lineNum = sentenceList[index][3]

        # below is comparing similarity of strings
        # if similarity above a threshold value 
        # remove bestSentence and move onto next iteration of while
        for sentence in report:
            sentence = sentence[0]
            if(similar(bestSentence, sentence) > 0.8):
                del sentenceList[index]
                continue
        

        numWords = getNumberOfWords(bestSentence)
        length += numWords
        report.append([bestSentence,lineNum])
        del sentenceList[index]

    # sort report list by 2nd column value 
    # this ranks list by which line in document they were found 
    # thus incorporating strutucal information to help produce coherent summary

    report.sort(key=itemgetter(1)) 

    report = [x[0] for x in report]


    #joins the sentences with a full stop and a space
    report = ". ".join(report)
    #needed for final full stop
    report = report + "."
    #hacky fix to clean up report
    report = report.replace("..", ".")
    report = report.replace("   ", " ")
    report = report.replace("  ", " ")
    return report
#------------------------------------------------------------------------------
# Section 2:    tf_idf

#Function to determine the number of documents a word appears in
def getNumAppears(word, matrix):
    timesAppear = 0;
    for i in range (0,len(termMatrix)):
        if word == termMatrix[i][0]:
            for x in range (1,number_of_files+1):
                if(termMatrix[i][x] != 0):
                    timesAppear += 1
            break
    return timesAppear

# Calculates the idf score for each word
def getIdfWeight(word,matrix):
    timesAppear = getNumAppears(word, matrix)
    # if word appears in all documents, idf = 0
    if(timesAppear == 0):
        idf = 0
    else:
        idf = math.log((number_of_files) / (timesAppear))
    return idf

#Calculate the tf-idf score for each word in each document.
def getTf_idf(word, matrix, column):
    idf = getIdfWeight(word, matrix)
    tf = getWordWeight(word, matrix, column)
    tf_idf = idf * tf
    return tf_idf
#------------------------------------------------------------------------------
#   Running methods


termMatrix = getTermMatrix()
sentenceList = getSentenceList(termMatrix)
sentenceList = locationInfo(sentenceList)
sentenceList = stuctureInfo(sentenceList)
#when the summary is <200 words, give longer sentence a reduced weighting
if (summaryLength < 101):
    sentenceList = shortSummary(sentenceList)

summary = getSummary(summaryLength, sentenceList, termMatrix)
print (summary)
