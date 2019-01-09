#! usr/bin/env python
# -*- coding: utf-8 -*-

# Toy IBM Model 1 
# Author: Emily Soward 
#
# An interactive and verbose implementation of Expectation Maximization 
# training and Viterbi decoding for simple lexical machine translation
# based on IBM Model 1.
#
# This module requests input at run-time specifying a corpus of source sentences
# and a corpus of target sentences, and will run until convergence of EM to 
# discover word alignments. The corpora must be paralell ordered lists of sentences.
#
# Output consists of the EM training iterations, learned probabilities per-word in 
# the source corpus, and sentence pair level alignments, which are Viterbi-decoded.

import re
from collections import defaultdict
import math

def cleanSource(corpus):
    '''Takes a list of read-in lines from one corpus, cuts off \n,
    and makes them into a list of tuples split on whitespace and
    given their length'''
    
    cleanSentences = []
    
    for i in corpus:
        i = i.lower()
        i = re.sub(r'\n*','', i) # strip out newlines in str
        i = re.split(r'\s*', i) # split on whitespace into a list
        i.insert(0, "NULL") # Add special token NULL to source only
        temp = tuple(i)
        
        cleanSentences.append(temp)

    return cleanSentences # is a list of tuples


def cleanTarget(corpus):
    '''Takes a list of read-in lines from one corpus, cuts off \n,
    and makes them into a list of tuples split on whitespace and
    given their length'''
    
    cleanSentences = []
    
    for i in corpus:
        i = i.lower()
        i = re.sub(r'\n*','', i) # strip out newlines in str
        i = re.split(r'\s*', i) # split on whitespace into a list
        temp = tuple(i)
        
        cleanSentences.append(temp)

    return cleanSentences # is a list of tuples

#################################


# pair(f,e) gets ALL possible alignments even if they're not seen
def pair(foreign, english):
    '''Takes two clean lists with sentences as tuples.
    Returns a dict of possible pairs of alignments in the parallel corpus.
    Pairs are all given the float value 0.0'''
    pairs = defaultdict(float)
    ftemp = ""
    etemp = ""

    #this is for every item in every tuple
    for i in foreign: # for every f sentence
        for j in i: # for every item in every f sentence
            ftemp = j
            for m in english: # every e sentence
                for n in m: # every item of every e sentence
                    etemp = n
                    temp= tuple([ftemp, etemp])
                    pairs[temp]
               
    return pairs


##################################

# initialize(f,e) sets all allignments t(e|f) uniformly. 
def initialize(foreign, english):
    
    '''Takes two clean lists with sentences as tuples.
    Returns a dict of possible pairs of alignments in the parallel corpus.
    Pairs are all given the float value 0.0'''

    funique = defaultdict(int)
    for i in foreign:
        for j in i:
            funique[j]

    initVal = 1/float(len(funique))

    
    init = pair(foreign, english)
    etemp = ""
    ftemp = ""

    for each in init:
        init[each] = initVal
        
               
    return init

###################################

def uniques(corpus):
    '''
    Counts unique words in corpus of sentences and returns them as a list.
    '''
    unDict = defaultdict(float)
    
    for i in corpus:
        for j in i:
            unDict[j]

    unList = unDict.keys()
    
    return unList


####################################

def em(foreign, english):

    MAX_ITERS = 1000
    MAX_DIFF = 1.0

    parallel = zip(foreign, english)
    probs = initialize(foreign, english)
    funiques = uniques(foreign)
    euniques = uniques(english)
    iters = 0
    

    while MAX_DIFF > 0.000001 and iters < MAX_ITERS:    
        count = pair(foreign, english)
        total = defaultdict(float)
        stots = defaultdict(float)

        for sentPair in parallel:

            for e in sentPair[1]:
                stots[e]
                for f in sentPair[0]:
                    stots[e] += probs.get(tuple([f,e]))

            for e in sentPair[1]:
                for f in sentPair[0]:
                    precalc = (probs.get(tuple([f,e]))) / stots.get(e)
                    count[tuple([f,e])] += precalc
                    total[f] += precalc
                    
        # Check convergence here
        local_max = 0.00
        difference = 0.0      
        for f in funiques:
            for e in euniques:
                difference = abs(probs[tuple([f,e])] - count.get(tuple([f,e])) / total.get(f))
                if difference > local_max:
                    local_max = difference
        print "Local Maximum: ", local_max
            
        MAX_DIFF = local_max

        
        for f in funiques:
            for e in euniques:
                probs[tuple([f,e])] = count.get(tuple([f,e])) / total.get(f)

        iters +=1
        print "EM Iteration ", iters


    return probs


#################################

def sortdict(dictionary):
    '''
    Sorts a dictionary by its values and returns a list of the items' keys
    sorted in declining order by value.
    '''
    values = dictionary.items()
    values.sort(key=lambda (k,v): (v,k))
    outlist = [key for key, value in values]
    outlist.reverse()
    
    return outlist

#################################

def transTable(probs):
    '''
    Takes the probabilities from em training and sorts it and prints a
    translation table. Outputs a dictionary used for viterbi.
    '''

    new = defaultdict(lambda: defaultdict(float))

    for k,v in probs.iteritems(): # this loop gives source keys
        new[k[0]][k[1]] = v

    myKeys = new.keys() # makes a list of all the words in the target
    myKeys.sort() # sorts the list of words in alphabetical order
    print myKeys

    tProb = float()
    sourceWord = ""
    targetWord = ""

    for word in myKeys:
        sourceWord = word
        # every new table gets a header
        print "\n{0}\n---------------------------".format(sourceWord) 
        for each in sortdict(new.get(word)):
            targetWord = each
            tProb = new[word].get(each)
            if tProb != 0.0: # restrict to non-zeros
            # all the non-zero items in the table get printed
                print "\tp({0}|{1}) = {2}".format(targetWord,sourceWord,tProb)


    return new # return the restructured dict for viterbi.


#################################

def viterbi(source, target, probs):

    para = zip(source, target)

    for sentPair in para:
        
        # Populate the trellis
        trellis = defaultdict(lambda: defaultdict(float))
        for s in sentPair[0]:
            trellis[s]
            for t in sentPair[1]:
                trellis[s][t] = probs[s].get(t)

        
        alignmentWords = []
        alignmentFormula = []
        fIndex = -1
        for s in sentPair[0]: # move through the target sentence in order
            fIndex += 1
            j = sortdict(trellis[s])
            alignmentWords.append(j[0])
            alignmentFormula.append("[{0}-->{1}]".format(j[0],fIndex))
            

        # Check to see if any english word aligns to NULL better than any
        # foreign word--don't output unless it does! 
        needNULL = None
        for each in alignmentWords[1:]:
            if alignmentWords[0] != each:
                needNULL = False
            else:
                needNULL = True

        # Print alignments appropriately if NULL token needed
        if needNULL:
            print "\n---------------------------"
            print "Sentence Pair:"
            print "\t" + " ".join(sentPair[0])
            print "\t" + " ".join(alignmentWords)
            print
            print "Alignment:"
            print "\t" + " ".join(alignmentFormula)
            print "(KEY: [target-->source index])"
        else: # doesn't need NULL token
            print "\n---------------------------"
            print "Sentence Pair:"
            print "\t" + " ".join(sentPair[0][1:])
            print "\t" + " ".join(alignmentWords[1:])
            print
            print "Alignment:"
            print "\t" + " ".join(alignmentFormula[1:])
            print "(KEY: [target-->source index])"

    return


##################################

def main():
    # Take in the stuff
    SOURCE = open(raw_input("<<Source-corpus filename>> "))
    TARGET = open(raw_input("<<Target-corpus filename>> "))
    rSOURCE = SOURCE.readlines()
    rTARGET = TARGET.readlines()
    SOURCE.close()
    TARGET.close()
    
    # Clean it appropriately
    my_source = cleanSource(rSOURCE)
    my_target = cleanTarget(rTARGET)

    # Use EM to calculate t(e|f)
    tProbs = em(my_source, my_target)
    # Get the translation tables
    tTables = transTable(tProbs)
    # Get the alignments
    alignments = viterbi(my_source, my_target, tTables)    

    return


if __name__ == '__main__':
    main()


