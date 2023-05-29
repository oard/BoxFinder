import PyPDF2
import os
import sys
import random
import json
import nltk
from nltk.stem.porter import *
import math
import html
import codecs
import re

prefix = 'C:/Users/oard/nara/BoxFinder/'
sums1 = [0, 0, 0]
sums2 = [0, 0, 0]

def getFiles(dir):
    dir = os.path.abspath(dir)
    pdfs = {}
    containers=[]
    first = True
    for root, subdirs, files in os.walk(dir):
        if first:
            first = False
        else:
            p=[]
            for filename in files:
                file_path = os.path.join(root, filename)
                if filename[-4:] != '.txt': p.append(filename)
            containers.append(root[33:38])
            pdfs[root[33:38]] = p
    return(containers, pdfs)

def selectFiles(pdfs, k):
    sampled = {}
    for pdf in pdfs:
        sampled[pdf] = random.sample(pdfs[pdf], k)
    return sampled

def selectQueries(containers,pdfs,sampled):
    sample = random.choices(containers, k=100)
    counts = {}
    answers = {}
    queryset = {}

    for cont in sample:
        if cont in counts:
            counts[cont] += 1
        else:
            counts[cont] = 1
    queries = set()
    for cont in counts:
        avail = set(pdfs[cont]).difference(set(sampled[cont]))
        q = random.sample(avail,counts[cont])
        queries = queries.union(q)
        queryset[cont] = q
        for query in q:
            answers[query] = cont
    return list(queries), queryset, answers

def buildCollection(sampled, n):
    first = True
    collection = {}
    for cont in sampled:
        fulltext = ""
        for doc in sampled[cont]:
            f=open(prefix+'pdf/'+cont+'/'+doc, 'rb')
            reader = PyPDF2.PdfReader(f)
            pages = len(reader.pages)
            for i in range(0,min(pages,n)):
                page = reader.pages[i]
                text = page.extract_text().replace('\n', ' ')
                if first:
#                    print(page.extract_text())
                    first = False
                fulltext = fulltext + text
#            print(f'Read {min(pages,n)} pages from file {doc} in Container {cont}')
        collection[cont] = fulltext
    return collection

def buildCollectionFromMetadata(sampled, short):
    metadata = readMetadata(short)
    collection = {}
    for cont in sampled:
        if cont in metadata:
            collection[cont] = metadata[cont]
        else:
            print('No metadata for container:', cont)
            exit(-1)
    length = 0
    for box in collection:
        length += len(collection[box])
    print('Boxes:', len(collection), 'Average words:', length/len(collection)/6)

    return collection

def countPages(pdfs):
    docs = 0
    total = 0
    for cont in pdfs:
        for doc in pdfs[cont]:
            docs += 1
            f = open(prefix + 'pdf/' + cont + '/' + doc, 'rb')
            reader = PyPDF2.PdfReader(f)
            pages = len(reader.pages)
            total += pages
#            print(doc, pages)
    print(f'Doucmments: {docs}, Total Pages: {total}, Average: {total/docs}')

def indexCollection(collection):
    stemmer = PorterStemmer()
#    nltk.download('punkt')  # This loads some NLTK tokenizers
    lengths = dict()
    index = dict()
    df = 0
    tf = 1
    i=0
    for cont in collection:
        body = collection[cont]
        tokens = nltk.word_tokenize(body)  # at this point the stems list actually contains unstemmed tokens
        lengths[cont] = len(tokens)
        for token in tokens:
            stem = stemmer.stem(token)
            if stem not in index: index[stem] = [0, dict()]
            if cont in index[stem][tf]:
                index[stem][tf][cont] += 1
            else:
                index[stem][df] += 1
                index[stem][tf][cont] = 1
    return index, lengths

def bm25_rank(query, index, lengths, debug=False):
#  df = 0
  tf = 1
  k1 = 1.2
  b = 0.75
  total_docs = len(lengths)
  total_len = 0
  for doc in lengths:
      total_len += lengths[doc]
  avg_len = total_len / total_docs
  stemmer = PorterStemmer()
  tokens = nltk.word_tokenize(query)
  bm25_score = dict()
  for i in range(len(tokens)):
    stem = stemmer.stem(tokens[i])
    if stem in index:
        dfvalue = len(index[stem][tf])
    else:
        dfvalue = 1
    idf_component = max(0,math.log((total_docs-dfvalue+0.5)/(dfvalue+0.5),2))
    if debug: print(f'\nTerm: "{stem}" DF: {dfvalue} BM-25 IDF component: {idf_component:.2f}\n')
    if stem in index:
        for doc in index[stem][tf]:
            tfvalue = index[stem][tf][doc]
            tf_component = ((k1+1)*tfvalue)/(k1*((1-b)+b*(lengths[doc]/avg_len))+tfvalue)
            if debug: print(f'Document: {doc:4} TF: {tfvalue:2} BM-25 TF component: {tf_component:.2f}')
            if doc in bm25_score: bm25_score[doc] += tf_component * idf_component
            else: bm25_score[doc] = tf_component * idf_component
  return {k: v for k, v in sorted(bm25_score.items(), key=lambda item: item[1], reverse=True)}

def readTitles():
    f = open(prefix+'titles.txt', 'r')
    lines = f.readlines()
    titles = {}
    for line in lines:
        file, title = line.split(' ', 1)
        if file+'.pdf' in titles:
            print('Found a dupe for:', file+'.pdf')
        titles[file+'.pdf'] = html.unescape(title.strip())
    return titles

def readFolderMetadata():
    f = open(prefix + 'folders.txt', 'r')
    lines = f.readlines()
    folders = {}
    for line in lines:
        file, folder = line.split(' ', 1)
        folders[file + '.pdf'] = html.unescape(folder.strip())
    return folders

def readCodes():
    f = open(prefix + 'codes.txt', 'r')
    lines = f.readlines()
    codes = {}
    for line in lines:
        file, code = line.split(' ', 1)
        codes[file + '.pdf'] = html.unescape(code.strip())
    return codes

def readMetadata(short=True):
    if short:
        f = open(prefix + 'foldermetadatashort.txt', 'r')
    else:
        f = open(prefix + 'foldermetadatalong.txt', 'r')
    lines = f.readlines()
    metadata = {}
    for line in lines:
        box, data = line.split('\t', 1)
        box = box.strip()
        if box in metadata:
            metadata[box] = metadata[box] + ' ' + data.strip()
        else:
            metadata[box] = data.strip()
    return metadata


def readQueryContent(queryset, n):
    first = True
    texts = {}
    for cont in queryset:
        fulltext = ""
        for doc in queryset[cont]:
            f=open(prefix+'pdf/'+cont+'/'+doc, 'rb')
            reader = PyPDF2.PdfReader(f)
            pages = len(reader.pages)
            for i in range(0,min(pages,n)):
                page = reader.pages[i]
                text = page.extract_text().replace('/n',' ')
                if first:
#                    print(text)
                    first = False
                fulltext = fulltext + text
            texts[doc] = fulltext
    return texts

def removeDate(s):
    nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero
    nMNTH = r'(?:11|12|10|0?[1-9])'  # month can be 1 to 12 with a leading zero
    nYR = r'(?:(?:19|20)\d\d)'
    nDELIM = r'(?:[\/\-\._])?'
#    NUM_DATE = f"""
#       (?:(?:{nYR}{nDELIM}{nMNTH}{nDELIM}{nDAY})|(?:{nYR}{nDELIM}{nDAY}{nDELIM}{nMNTH})|(?:{nDAY}{nDELIM}{nMNTH}{nDELIM}{nYR})|(?:{nMNTH}{nDELIM}{nDAY}{nDELIM}{nYR})
#        )"""
    NUM_DATE = r'(?:11|12|10|0?[1-9])(?:[\/\-\._])(?:[0-3]?\d)(?:[\/\-\._])(?:(?:19|20)?\d\d)'
    pattern =  re.compile(NUM_DATE, re.IGNORECASE | re.VERBOSE | re.UNICODE)
    match = pattern.search(s)
    if match:
        return s.replace(match.group(),'').strip()
    else:
        return s
#    return re.search(pattern, s).group()

def runQueries(queries, queryset, index, lengths, answers, titleQueries, n):
    if titleQueries:
        queryText = readTitles()
    else:
        queryText = readQueryContent(queryset, n)
    results = {}
    correct = 0
    near = 0
    rank2 = 0
    for query in queries:
        if query in queryText:
            results[query] = bm25_rank(queryText[query].lower(), index, lengths)
            rankedList = list(results[query].keys())
            if len(rankedList)>0:
                if rankedList[0] == answers[query]:
                    correct += 1
                elif len(rankedList)>1 and rankedList[1] == answers[query]:
                    rank2 += 1
                else:
                    diff = abs(int(rankedList[0]) - int(answers[query]))
                    if diff == 1: near += 1
#                    print(f'Bad answer {rankedList[0]} when correct answer was {answers[query]} for difference of {diff}')
            else:
                print(f'No ranked list for query {query}')
        else:
            print(f'Query {query} not found in query text')
    return results, correct, near, rank2

def mergeResults(results1, results2, answers):
    results = {}
    for query in answers:
        if query in results1 and query in results2:
            rr1={}
            list1 = list(results1[query].keys())
#            print(list1)
            for i in range(len(list1)):
                rr1[list1[i]] = 1/(i+61)
            rr2={}
            list2 = list(results2[query].keys())
            for i in range(len(list2)):
                rr2[list2[i]] = 1/(i+61)
            for doc in rr1:
                if doc in rr2:
                    rr1[doc] += rr2[doc]
            for doc in rr2:
                if doc not in rr1:
                    rr1[doc] = rr2[doc]
            results[query] = dict(sorted(rr1.items(), reverse=True, key=lambda x: x[1]))
        elif query in results1:
            results[query] = results1[query]
        elif query in results2:
            results[query] = results2[query]
        else:
            results[query] = {}

#    results = results1

    correct = 0
    near = 0
    rank2 = 0
    for query in results:
        rankedList = list(results[query].keys())
        if len(rankedList) > 0:
            if rankedList[0] == answers[query]:
                correct += 1
            elif len(rankedList) > 1 and rankedList[1] == answers[query]:
                rank2 += 1
            else:
                diff = abs(int(rankedList[0]) - int(answers[query]))
                if diff == 1:
                    near += 1
        else:
            print(f'No ranked list for query {query} DURING MERGE' )
#    print('Results 1:',results1) # ['352415.pdf'])
#    print('Results 2:', results2) #['352415.pdf'])
#    print('Results:', results[list(results1.keys())[0]]) # ['352415.pdf'])
    return results, correct, near, rank2

def runExperiment(titleQueries, maxpages, samples, g, h, scans = True):
    both = False
    containers, pdfs = getFiles(prefix+'pdf/')
#    countPages(pdfs)
    sampled = selectFiles(pdfs, samples)
    queries, queryset, answers = selectQueries(containers, pdfs, sampled)
    #    queries.append(sampled['1900'][0]) # testing the dupe detection code
    for query in queries:
        for cont in sampled:
            for samp in sampled[cont]:
                if query == samp: print('Bad dupe found', query)
    if both:
        collection1 = buildCollection(sampled, maxpages)
        index1, lengths1 = indexCollection(collection1)
        results1, correct1, close1, rank21 = runQueries(queries, queryset, index1, lengths1, answers, titleQueries, maxpages)
        sums1[0] += correct1
        sums1[1] += close1
        sums1[2] += rank21
#        print(f'Max Pages: {maxpages}, Samples: {samples}, Correct: {correct1:2}, Close: {close1:2}, Rank 2: {rank21:2}',file=g)
        collection2 = buildCollectionFromMetadata(sampled, short=True)
        index2, lengths2 = indexCollection(collection2)
        results2, correct2, close2, rank22 = runQueries(queries, queryset, index2, lengths2, answers, titleQueries, maxpages)
        sums2[0] += correct2
        sums2[1] += close2
        sums2[2] += rank22
#        print(f'Max Pages: {maxpages}, Samples: {samples}, Correct: {correct2:2}, Close: {close2:2}, Rank 2: {rank22:2}', file=h)
        results, correct, close, rank2 = mergeResults(results1, results2, answers)
    elif scans:
        collection = buildCollection(sampled, maxpages)
        index, lengths = indexCollection(collection)
        results, correct, close, rank2 = runQueries(queries, queryset, index, lengths, answers, titleQueries, maxpages)
    else:
        collection = buildCollectionFromMetadata(sampled, short=False)
        index, lengths = indexCollection(collection)
        results, correct, close, rank2 = runQueries(queries, queryset, index, lengths, answers, titleQueries, maxpages)
    return correct, close, rank2

def runExperimentSet(titleQueries, maxpagesValues, samplesValues, runs, scans=True):
#    titleQueries = False
#    maxpagesValues = [1, 2, 3, 4, 100]
#    samplesValues = [1, 2, 3, 4, 6, 8, 10]
#    runs = 25

    sums = list([None] * (max(maxpagesValues) + 1))
    f = open(prefix + 'results.txt', 'w')
    h = open(prefix + 'results-foldermetadata.txt', 'w')
    g = open(prefix + 'results-sampledscans.txt', 'w')
    print(f'Title Queries: {titleQueries}')
    for maxpages in maxpagesValues:
        sums[maxpages] = list([None] * (max(samplesValues) + 1))
        for samples in samplesValues:
            sums[maxpages][samples] = [0, 0, 0]
            for i in range(0, 3):
                sums1[i] = 0
                sums2[i] = 0
            for run in range(0, runs):
                correct, close, rank2 = runExperiment(titleQueries, maxpages, samples, g, h, scans)
                sums[maxpages][samples][0] += correct
                sums[maxpages][samples][1] += close
                sums[maxpages][samples][2] += rank2
                print(
                    f'Max Pages: {maxpages}, Samples: {samples}, Run: {run + 1}, Correct: {correct:2}, Close: {close:2}, Rank 2: {rank2:2}')
            #                print(f'Max Pages: {maxpages}, Samples: {samples}, Run: {run+1}, Correct: {correct:2}, Close: {close:2}, Rank 2: {rank2:2}',file=f)
            for i in range(0, 3):
                sums[maxpages][samples][i] = sums[maxpages][samples][i] / runs
                sums1[i] = sums1[i] / runs
                sums2[i] = sums2[i] / runs
            print(
                f'Max Pages: {maxpages}, Samples: {samples}, Runs: {runs}, Correct: {sums[maxpages][samples][0]:.1f}, Close: {sums[maxpages][samples][1]:.1f}, Rank 2: {sums[maxpages][samples][2]:.1f}')
            print(
                f'Max Pages: {maxpages}, Samples: {samples}, Runs: {runs}, Correct: {sums[maxpages][samples][0]:.1f}, Close: {sums[maxpages][samples][1]:.1f}, Rank 2: {sums[maxpages][samples][2]:.1f}',
                file=f)
            print(
                f'Max Pages: {maxpages}, Samples: {samples}, Runs: {runs}, Correct: {sums1[0]:.1f}, Close: {sums1[1]:.1f}, Rank 2: {sums1[2]:.1f}',
                file=g)
            print(
                f'Max Pages: {maxpages}, Samples: {samples}, Runs: {runs}, Correct: {sums2[0]:.1f}, Close: {sums2[1]:.1f}, Rank 2: {sums2[2]:.1f}',
                file=h)
    f.close()
    g.close()
    h.close()

if __name__ == '__main__':

    runExperimentSet(titleQueries=True, maxpagesValues=[1], samplesValues=[1], runs=4200, scans=False)

    exit(2)

    titles = readTitles()
    folders = readFolderMetadata()
    codes = readCodes()


    found = 0
    foundcode = 0
    notfound = 0
    snc={}
    for doc in titles:
        neither = True
        if doc in folders:
#            print('Found:', doc, folders[doc], titles[doc])
            found += 1
#            snc[doc] = folders[doc]
#            print(snc[doc], '|', removeDate(snc[doc]))
            snc[doc] = removeDate(folders[doc])
            neither = False
#        else:
#            print('Not found in titles:', doc, titles[doc])
        if doc in codes:
            #            print('Found:', doc, folders[doc], titles[doc])
            foundcode += 1
            snc2 = codes[doc].replace('XR ', '')
            if (not neither) and snc2 not in snc[doc]:
#                print('MISMATCH: Folder SNC:', snc[doc], 'Metadata SNC:', snc2, 'Full Folder Name:', folders[doc])
                print('Folder:', snc[doc], '\t\tMetadata:', snc2)
            snc[doc] = snc2
            neither = False
#        else:
#            print('Not found in codes:', doc, titles[doc])
        if neither:
            notfound += 1
            print('Neither in folders nor codes: doc, titles[doc]')
    print('Total folders found:', found, 'Total Codes Found', foundcode, 'Total docs not found in either:', notfound)

    print('\nUnique Subject-Numeric Codes')
    unique = {}
    for doc in snc:
        if snc[doc] not in unique:
            print(snc[doc])
            unique[snc[doc]] = 'TBA'

