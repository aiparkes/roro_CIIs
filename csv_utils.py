import csv
import numpy as np
import utils
import sys

#functions which read or write to csv

def makeDictNames(file):
    #makes names 'dict_0', 'dict_1'...'dict_X' for X number of dicts in list in file
    with open(file, 'rU', encoding='utf-8-sig') as f:
        w = csv.reader(f,delimiter=',')
        length = sum(1 for row in w)
        dictNames = ['dict_'+str(num) for num in np.arange(length)]
    return(dictNames)

def readListOfDictsFromCsv(file):
    #reads list of dictionaries from a csv
    #assumes a list with headers at the top, not repeated
    #reads rows in one by one and uses readOneDict to return lists of keys and items
    #updates these to temp_dict, then temp_dict to motherDict
    motherDict = {}
    dictNames = makeDictNames(file)
    with open(file, 'rU', encoding='utf-8-sig') as f:
        w = csv.reader(f,delimiter=',')
        count = 0
        for indx, row in enumerate(w):
            if indx==0:
                raw_headers = (0,row)
            else:
                raw_values = (1,row)
                headers, values = readOneDict([raw_headers,raw_values])
                temp_dict = {}
                for header, value in zip(headers,values):
                    temp_dict.update({header:value})
                motherDict.update({dictNames[count]:temp_dict})
                count+=1
    return(motherDict)

def readOneDict(enum_w):
    #takes an enumerated list of headers and values
    #for header/row appends all elements to new lists, returns lists
    headers = []
    values = []
    for indx, row in enum_w:
        if indx==0:
            for element in row:
                headers.append(element)
        else:
            for element in row:
                values.append(element)
    return(headers, values)

def readVecFromCsv(file):
    with open(file, 'rU', encoding='utf-8-sig') as f:
        w = csv.reader(f, delimiter = ',')
        output = []
        for row in w:
            output.append(row)
        return row

def readDictFromCsv(file):
    #reads one dict from a file (csv)
    temp_dict = {}
    with open(file, 'rU', encoding='utf-8-sig') as f:
        w = csv.reader(f, delimiter=',')
        headers, values = readOneDict(enumerate(w))
        for header, value in zip(headers,values):
            temp_dict.update({header:value})
    return temp_dict

@utils.incaseCsvOpen
def writeDictToCsv(dict, header=True, fileName='someFile.csv'):
    #writes dictionary to fileName (csv)
    #if header also write keys as first line
    with open(fileName, 'a', newline='\n') as f:
        w = csv.DictWriter(f, dict.keys())
        if header:
            w.writeheader()
        w.writerow(dict)
    return

@utils.incaseCsvOpen
def writeMatrixToCsv(matrix, fileName='someFile.csv'):
    #writes matrix to fileName (csv)
    with open(fileName, 'a', newline='\n') as f:
        w = csv.writer(f)
        for row in matrix:
            w.writerow(row)
    return

@utils.incaseCsvOpen
def writeVectorToCsv(vector,fileName='someFile.csv'):
    #writes vector to fileName (csv)
    with open(fileName, 'a', newline='\n') as f:
        w = csv.writer(f)
        w.writerow(vector)
    return

if __name__ == '__main__':
    main()
