import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import copy
import os

#import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.special import softmax

#from classicDynamic import *

from tqdm import tqdm




class TransitionModel(nn.Module):
    def __init__(self, Nstate):
        super(TransitionModel, self).__init__()

        
        
        self.transRight = torch.nn.Parameter(torch.rand( (Nstate * 4, Nstate) ).float())
        self.transLeft = torch.nn.Parameter(torch.rand( (Nstate * 4, Nstate) ).float())
        self.transPair = torch.nn.Parameter(torch.rand( (Nstate * 4 * 4, Nstate) ).float())
        
        #self.transCombine = torch.nn.Parameter(torch.rand( (Nstate * Nstate, Nstate) ).float())
        Nstate2 = Nstate // 2
        #Nstate2 = 2 #Ultra simplified 2 state version
        #Nstate2 = Nstate
        self.transCombine = torch.nn.Parameter(torch.rand( (Nstate2 * Nstate2, Nstate) ).float())

        self.divider = Nstate // Nstate2



    def addRight(self, x, nucVector):
        matrix1 = F.log_softmax(self.transRight, dim=1)
        matrix1 = matrix1.reshape((1, matrix1.shape[0], matrix1.shape[1]))

        tupleProbs = x.reshape((x.shape[0], x.shape[1], 1)) + nucVector.reshape((nucVector.shape[0], 1, nucVector.shape[1]))
        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1]*tupleProbs.shape[2]))
        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1], 1))
        tupleProbs = torch.logsumexp(tupleProbs + matrix1, dim=1)

        #x = x.reshape((x.shape[0], x.shape[1], 1))
        #x = torch.logsumexp(x + matrix1, dim=1)

        return tupleProbs
    
    def addLeft(self, x, nucVector):
        matrix1 = F.log_softmax(self.transLeft, dim=1)
        matrix1 = matrix1.reshape((1, matrix1.shape[0], matrix1.shape[1]))


        tupleProbs = x.reshape((x.shape[0], x.shape[1], 1)) + nucVector.reshape((nucVector.shape[0], 1, nucVector.shape[1]))
        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1]*tupleProbs.shape[2]))
        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1], 1))
        tupleProbs = torch.logsumexp(tupleProbs + matrix1, dim=1)


        #x = x.reshape((x.shape[0], x.shape[1], 1))
        #x = torch.logsumexp(x + matrix1, dim=1)
        
        return tupleProbs

    def addPair(self, x, nucVector):
        
        
        matrix1 = self.transPair #matrix1 = F.log_softmax(self.transPair, dim=1) #Commented to allow for increases
        matrix1 = matrix1.reshape((1, matrix1.shape[0], matrix1.shape[1]))
        
        #matrix1 = torch.zeros((4, 4))
        #matrix1[1, 3] = 20
        #matrix1[3, 1] = 20
        #matrix1[0, 2] = 20
        #matrix1[2, 0] = 20
        #matrix1 = matrix1.reshape((16, 1))

        #print ("mat")
        #print (matrix1)
        #print (nucVector)



        #x = x.reshape((x.shape[0], x.shape[1], 1))
        tupleProbs = x.reshape((x.shape[0], x.shape[1], 1)) + nucVector.reshape((nucVector.shape[0], 1, nucVector.shape[1]))
        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1]*tupleProbs.shape[2]))

        #print (tupleProbs.shape)
        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1], 1))

        tupleProbs = torch.logsumexp(tupleProbs + matrix1, dim=1)

        #print (tupleProbs)


        #x = torch.logsumexp(x + matrix1, dim=1)
        return tupleProbs
    

    def doCombine(self, x, y):

        if True:
            #div1 = self.divider
            div1 = 2
            x = x.reshape((x.shape[0], x.shape[1] // div1, div1))
            x = torch.logsumexp(x, axis=2)
            y = y.reshape((y.shape[0], y.shape[1] // div1, div1))
            y = torch.logsumexp(y, axis=2)
        
        tupleProbs = x.reshape((x.shape[0], x.shape[1], 1)) + y.reshape((y.shape[0], 1, y.shape[1]))
        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1]*tupleProbs.shape[2]))

        matrix1 = F.log_softmax(self.transCombine, dim=1)
        matrix1 = matrix1.reshape((1, matrix1.shape[0], matrix1.shape[1]))

        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1], 1))
        #print (x.shape)
        #print (matrix1.shape)
        tupleProbs = torch.logsumexp(tupleProbs + matrix1, dim=1)

        return tupleProbs
    

    def fastCombine(self, x, y):

        if True:
            #div1 = self.divider
            div1 = 2
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2] // div1, div1))
            x = torch.logsumexp(x, axis=3)
            y = y.reshape((y.shape[0], y.shape[1], y.shape[2] // div1, div1))
            y = torch.logsumexp(y, axis=3)

        #print (x.shape)
        #quit()

        

        tupleProbs = x.reshape((x.shape[0],  x.shape[1], x.shape[2], 1)) + y.reshape((y.shape[0], y.shape[1], 1, y.shape[2]))
        tupleProbs = torch.logsumexp(tupleProbs, axis=1)
        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1]*tupleProbs.shape[2]))
        


        matrix1 = F.log_softmax(self.transCombine, dim=1)
        matrix1 = matrix1.reshape((1, matrix1.shape[0], matrix1.shape[1]))

        tupleProbs = tupleProbs.reshape((tupleProbs.shape[0], tupleProbs.shape[1], 1))

        #print (tupleProbs.shape)
        
        tupleProbs = torch.logsumexp(tupleProbs + matrix1, dim=1)

        

        return tupleProbs
    
    def quickOverestimateCombiine(self, x, y):


        sumLogX = torch.logsumexp(x, axis=2)
        sumLogY = torch.logsumexp(y, axis=2)
        relXmax = torch.max(x - sumLogX.reshape((x.shape[0], x.shape[1], 1)), axis=1)[0]
        relYmax = torch.max(y - sumLogY.reshape((x.shape[0], x.shape[1], 1)), axis=1)[0]
        sumLogMax = torch.max(sumLogX + sumLogY, axis=1)

        sumLog = torch.logsumexp(x, axis=2) + torch.logsumexp(y, axis=2)

        relXmax = relXmax + sumLog
        maxArray = relXmax.reshape((relXmax.shape[0], relXmax.shape[1], 1)) + relYmax.reshape((relYmax.shape[0], 1, relYmax.shape[1]))

        maxArray = maxArray.reshape((maxArray.shape[0], maxArray.shape[1]*maxArray.shape[2]))
        #print (maxArray.shape)

        matrix1 = F.log_softmax(self.transCombine, dim=1)
        matrix1 = matrix1.reshape((1, matrix1.shape[0], matrix1.shape[1]))
        maxArray = maxArray.reshape((maxArray.shape[0], maxArray.shape[1], 1))

        maxArray = torch.logsumexp(maxArray + matrix1, dim=1)

        return maxArray




    def forward(self, x):

        print ('hi')

        return x
    



class coefModel(torch.nn.Module):
    def __init__(self, coef_initial):
        super(coefModel, self).__init__()
        self.coef = torch.nn.Parameter(torch.tensor(coef_initial).float())

    def forward(self):
        return torch.abs(self.coef)




def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data



def saveAmino():

    file1 = './data/aminoData/codon.txt'

    data = np.loadtxt(file1, dtype=str)

    data = data[data[:, 3] != 'Stop']

    data2 = np.zeros((  data.shape[0], 4 ), dtype=int)

    _, aminoAcid_inverse = np.unique(data[:, 2], return_inverse=True)
    data2[:, 0] = aminoAcid_inverse

    nucleotides = ['A', 'C', 'G', 'T']
    dictNuc = {}
    for a in range(len(nucleotides)):
        dictNuc[nucleotides[a]] = a

    for a in range(data.shape[0]):

        codon = data[a, 0]

        nuc1 = list(codon)

        nuc1[0] = dictNuc[nuc1[0]]
        nuc1[1] = dictNuc[nuc1[1]]
        nuc1[2] = dictNuc[nuc1[2]]

        data2[a, 1] = nuc1[0]
        data2[a, 2] = nuc1[1]
        data2[a, 3] = nuc1[2]

    #print (data2)

    np.save('./data/aminoData/numberCodon.npy',  data2)

#saveAmino()
#quit()


def OLD_saveRealData():

    folder1 = './data/realData/published_rdat/'
    files1 = os.listdir(folder1)

    data = []

    for a in range(len(files1)):
        filename1 = files1[a]

        #print (filename1)

        try:
            file1 = open(folder1 + filename1, 'r')
            Lines = file1.readlines()

            seqLine = Lines[2]
            structureLine = Lines[3]

            seqLine = seqLine.replace('\n', '')
            structureLine = structureLine.replace('\n', '')

            sequence = seqLine.split('\t')[1]
            structure = structureLine.split('\t')[1]

            validStructure = '(' in list(structure)
            validSequence = not 'X' in list(sequence)

            if validStructure and validSequence:

                data.append([ filename1, sequence, structure ])
                

        except:
            True

    
    data = np.array(data)

    np.save('./data/realData/published_rdat.npy', data)

    #print (files1)

#OLD_saveRealData()
#quit()


def saveReadData():

    folder1 = './data/realData/dbnFiles/'
    files1 = os.listdir(folder1)

    data = []

    for a in range(len(files1)):
        filename1 = files1[a]

        #print (filename1)

        file1 = open(folder1 + filename1, 'r')
        Lines = file1.readlines()

        length1 = Lines[1]
        #print ([length1])
        length1 = length1[len('#Length:'):]
        length1 = length1.replace('\n', '')
        length1 = length1.replace(' ', '')
        length1 = length1.replace(',', '')
        
        length1 = int(length1)

        if length1 <= 50:

            sequence = Lines[3]
            structure = Lines[4]

            sequence = sequence.replace('\n', '')
            structure = structure.replace('\n', '')

            data.append([sequence, structure])

        print (a, len(data), len(files1))
        
        

    
    data = np.array(data)

    np.savez_compressed('./data/realData/dbnFiles_short50.npz', data)

    #print (files1)

#saveReadData()
#quit()



def checkConvertSequence(sequence):

    list1 = ['A', 'C', 'G', 'U']
    list2 = ['a', 'c', 'g', 'u']

    for a in range(len(list1)):
        sequence[sequence==list1[a]] = a
        sequence[sequence==list2[a]] = a

    sequence = sequence.astype(int)

    return sequence



def makeMatrix(inputStrings):

    matrix1 = np.zeros((inputStrings.shape[0], length1, length1), dtype=int)
    for a in range(inputStrings.shape[0]):
        strings1 = inputStrings[a]
        number1 = np.argwhere(strings1[:, 0] == -1)[0]
        strings1 = strings1[:number1]

        matrix1[a, strings1[:, 0], strings1[:, 1] ] = 1
    
    return matrix1


def torch_logsumexp(x, y):

    if len(x.shape) == 1:
        x = x.reshape((1, x.shape[0]))
        y = y.reshape((1, y.shape[0]))
    if len(x.shape) == 2:
        x = x.reshape((1, x.shape[0], x.shape[1]))
        y = y.reshape((1, y.shape[0], y.shape[1]))
    if len(x.shape) == 3:
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2]))
    
    result1 = torch.logsumexp(torch.cat((x, y), axis=0), axis=0)
    return result1


def simpleToSequenceMatrix(simpleSequencePair):

    structureMatrix = np.zeros((simpleSequencePair.shape[0], simpleSequencePair.shape[0]), dtype=int)

    unique1 = np.unique(simpleSequencePair)
    unique1 = unique1[unique1!=0]

    for val1 in unique1:
        args1 = np.argwhere(simpleSequencePair == val1)[:, 0]
        structureMatrix[args1[0], args1[1]] = 1
        structureMatrix[args1[1], args1[0]] = 1

    return structureMatrix

def getStructureArray(structure):
    
    structureCopy = copy.deepcopy(structure)

    count1 = 1
    continue1 = True 
    while ('(' in structureCopy):
        

        sum1 = 0
        a = 0
        while structureCopy[a] != '(':
            a += 1
        
        sum1 = 1
        b = a #+ 1
        while sum1 != 0:
            b += 1
            if structureCopy[b] == '(':
                sum1 += 1
            if structureCopy[b] == ')':
                sum1 -= 1
            
        
        structureCopy[a] = count1
        structureCopy[b] = count1
        
        count1 += 1

    for a in range(len(structureCopy)):
        if structureCopy[a] == '.':
            structureCopy[a] = 0
    structureCopy = np.array(structureCopy)

    return structureCopy


def structureToMatrix(structure):

    structureCopy = getStructureArray(structure)

    #print (structureCopy)
    #quit()
    matrix1 = simpleToSequenceMatrix(structureCopy)

    return matrix1


def evaluateAccuracy(structure_true, structure_pred):

    upperTriangle = np.eye(len(structure_true))
    upperTriangle = np.cumsum(upperTriangle, axis=0)
    upperTriangle[upperTriangle >= 1] = 1


    matrix_true = structureToMatrix(structure_true)
    matrix_pred = structureToMatrix(structure_pred)

    matrix_true = matrix_true * upperTriangle
    matrix_pred = matrix_pred * upperTriangle



    error1 = np.sum(np.abs(matrix_true - matrix_pred))
    truePair = np.sum(matrix_true)

    return error1, truePair





def countStructure(structure):
    structureCopy = getStructureArray(structure)

    unique1 = np.unique(structureCopy)
    unique1 = unique1[unique1!=0]

    elements = np.zeros(5)

    for a in range(unique1.shape[0]):

        args1 = np.argwhere(structureCopy == unique1[a])[:, 0]


        pos1 = args1[0] + 1
        pos2 = args1[1] - 1

        while structureCopy[pos1] == 0:
            pos1 += 1
        while structureCopy[pos2] == 0:
            pos2 -= 1

        #print ('')
        #print (unique1[a])
        #print (structureCopy)
        #print (args1)
        #print (pos1, pos2)
        
        if structureCopy[pos1] == unique1[a]:
            elements[0] += 1
        elif structureCopy[pos1] != structureCopy[pos2]:
            elements[4] += 1
        else:

            if pos1 == args1[0] + 1:

                if pos2 == args1[1] - 1:
                    elements[1] += 1
                else:
                    elements[2] += 1
            
            else:

                if pos2 == args1[1] - 1:
                    elements[2] += 1
                else:
                    elements[3] += 1


    #print (elements)
    #quit()

    return elements










    print (structureCopy)
    quit()



def backtraceNeural(sequence, Nstate, Nsample, filledTable, model):
    

    queList = np.zeros((sequence.shape[0]*2, Nsample, 2), dtype=int) - 1
    valuations = np.zeros((sequence.shape[0]*2, Nsample, Nstate), dtype=float)
    
    queList[0, :, 0] = 0
    queList[0, :, 1] = sequence.shape[0] - 1

    structureString = np.zeros((sequence.shape[0], Nsample), dtype=int).astype(str)
    structureString[:] = '.'


    
    for length0 in range(0, sequence.shape[0] - 1): #Min lenght is 2
        length1 = sequence.shape[0] - length0

        lengthAll = (queList[:, :, 1] - queList[:, :, 0]) + 1
        argLength = np.argwhere(lengthAll == length1)

        if argLength.shape[0] > 0:

            valuations_now = valuations[argLength[:, 0], argLength[:, 1]]



            start1 = queList[argLength[:, 0], argLength[:, 1], 0]
            end1 = queList[argLength[:, 0], argLength[:, 1], 1]


            #if start1[0] == 2:
            #    print (start1[0], end1[0])


            #Add pair    
            tableChunk = filledTable[start1+1, end1-1]
            nuc1 = sequence[start1]
            nuc2 = sequence[end1]
            nucVector = np.zeros((start1.shape[0], 4, 4), dtype=int) - 5000
            nucVector[np.arange(start1.shape[0]), nuc1, nuc2] = 0
            nucVector = nucVector.reshape((start1.shape[0], 16))
            nucVector = torch.tensor(nucVector).float()

            newStates_pair = model.addPair(tableChunk, nucVector).data.numpy()



            
            
            #Add left
            nuc1 = sequence[start1]
            nucVector = np.zeros((start1.shape[0], 4), dtype=int) - 5000
            nucVector[np.arange(start1.shape[0]), nuc1] = 0
            nucVector = torch.tensor(nucVector).float()

            tableChunk = filledTable[start1+1, end1]
            newStates_left = model.addLeft(tableChunk, nucVector).data.numpy()
            #filledTable[start1, end1] = torch_logsumexp( filledTable[start1, end1] , newStates  )
            
            
            #Add right
            nuc2 = sequence[end1]
            nucVector = np.zeros((start1.shape[0], 4), dtype=int) - 5000
            nucVector[np.arange(start1.shape[0]), nuc2] = 0
            nucVector = torch.tensor(nucVector).float()

            tableChunk = filledTable[start1, end1-1]
            newStates_right = model.addRight(tableChunk, nucVector).data.numpy()
            #filledTable[start1, end1] = torch_logsumexp( filledTable[start1, end1] , newStates  )

            


            #option 4, combine two pieces
            if length1 >= 3:

                newStates_combine = np.zeros((length1 - 2, start1.shape[0], Nstate ))

                for adder1 in range(length1 - 2):
                    end2 = start1 + 1 + adder1
                    start2 = end2 + 1
                    tableChunk1 = filledTable[start1, end2]
                    tableChunk2 = filledTable[start2, end1]
                    newStates = model.doCombine(tableChunk1, tableChunk2).data.numpy()
                    newStates_combine[adder1] = np.copy(newStates)
                    #filledTable[start1, end1] = torch_logsumexp ( filledTable[start1, end1], newStates)

            else:

                newStates_combine = np.zeros((1, start1.shape[0], Nstate )) - 5000


            newStates_pair = newStates_pair.reshape((1, newStates_pair.shape[0], newStates_pair.shape[1]))
            newStates_left = newStates_left.reshape((1, newStates_left.shape[0], newStates_left.shape[1]))
            newStates_right = newStates_right.reshape((1, newStates_right.shape[0], newStates_right.shape[1]))
            newStates_cat = np.concatenate(( newStates_pair , newStates_left, newStates_right, newStates_combine  ), axis=0)

            #if start1[0] == 2:
            #    print ("B")
            #    print (newStates_pair[0, 0])
            #    print (newStates_left[0, 0])
            #    print (newStates_right[0, 0])

            valuations_reshape = valuations_now.reshape((1, valuations_now.shape[0], valuations_now.shape[1]))


            newStates_weighted = logsumexp(newStates_cat + valuations_reshape, axis=2)

            


            newStates_weighted_exp = softmax(newStates_weighted, axis=0)
            choices = np.zeros(newStates_weighted_exp.shape[1], dtype=int)
            for choiceIndex in range(choices.shape[0]):
                choice1 = np.random.choice(newStates_weighted.shape[0], size=1, replace=True, p=newStates_weighted_exp[:, choiceIndex])[0]
                choices[choiceIndex] = choice1 

            
            #if start1[0] == 2:
            #    print (choices[0])
            #    quit()

            #print (choices.shape)
            #print (valuations_now.shape)
            #print (argLength.shape)
            for choiceIndex in range(choices.shape[0]):
                choice1 = choices[choiceIndex]
                currentValuation = valuations_now[choiceIndex]

                #print (currentValuation.shape)
                #quit()

                #print (choice1)
                sampleNum = argLength[choiceIndex][1]
                nextQue = np.argwhere(queList[:, sampleNum, 0] == -1)[0, 0]


                if choice1 == 0:

                    queList[nextQue, sampleNum, 0] = start1[choiceIndex]+1
                    queList[nextQue, sampleNum, 1] = end1[choiceIndex]-1
                    
                    
                    structureString[start1[choiceIndex], sampleNum] = '('
                    structureString[end1[choiceIndex], sampleNum] = ')'
                    
                    nuc1 = sequence[start1]
                    nuc2 = sequence[end1]
                    nucVector = np.zeros((1, 4, 4), dtype=int) - 5000
                    nucVector[0, nuc1, nuc2] = 0
                    nucVector = nucVector.reshape((1, 16))
                    nucVector = torch.tensor(nucVector).float()

                    newValueVector = []
                    for valIndex in range(Nstate):      
                        tableChunk = torch.zeros((1, Nstate)) - 5000
                        tableChunk[0, valIndex] = 0
                        valuesMini = model.addPair(tableChunk, nucVector).data.numpy()
                        #print (valuesMini)
                        valuesMini = logsumexp(valuesMini[0] + currentValuation )
                        newValueVector.append(valuesMini)
                    #quit()

                    newValueVector = np.array(newValueVector)
                    valuations[nextQue, sampleNum] = newValueVector

                if choice1 == 1:

                    queList[nextQue, sampleNum, 0] = start1[choiceIndex]+1
                    queList[nextQue, sampleNum, 1] = end1[choiceIndex]

                    
                    nuc1 = sequence[start1[choiceIndex]]
                    nucVector = np.zeros((1, 4), dtype=int) - 5000
                    nucVector[0, nuc1] = 0
                    nucVector = torch.tensor(nucVector).float()



                    newValueVector = []
                    for valIndex in range(Nstate):      
                        tableChunk = torch.zeros((1, Nstate)) - 5000
                        tableChunk[0, valIndex] = 0
                        valuesMini = model.addLeft(tableChunk, nucVector).data.numpy()
                        #print (valuesMini)
                        valuesMini = logsumexp(valuesMini[0] + currentValuation )
                        newValueVector.append(valuesMini)
                    #quit()
                    newValueVector = np.array(newValueVector)
                    valuations[nextQue, sampleNum] = newValueVector

            
                if choice1 == 2:



                    queList[nextQue, sampleNum, 0] = start1[choiceIndex]
                    queList[nextQue, sampleNum, 1] = end1[choiceIndex]-1
                    

                    nuc2 = sequence[end1[choiceIndex]]
                    nucVector = np.zeros((1, 4), dtype=int) - 5000
                    nucVector[0, nuc2] = 0
                    nucVector = torch.tensor(nucVector).float()



                    newValueVector = []
                    for valIndex in range(Nstate):      
                        tableChunk = torch.zeros((1, Nstate)) - 5000
                        tableChunk[0, valIndex] = 0
                        valuesMini = model.addRight(tableChunk, nucVector).data.numpy()
                        valuesMini = logsumexp(valuesMini[0] + currentValuation )
                        newValueVector.append(valuesMini)

                    newValueVector = np.array(newValueVector)
                    valuations[nextQue, sampleNum] = newValueVector

                
                if choice1 >= 3:

                    adder1 = choice1 - 3
                    end2 = start1[choiceIndex] + 1 + adder1
                    start2 = end2 + 1

                    queList[nextQue, sampleNum, 0] = start1[choiceIndex]
                    queList[nextQue, sampleNum, 1] = end2
                    queList[nextQue+1, sampleNum, 0] = start2
                    queList[nextQue+1, sampleNum, 1] = end1[choiceIndex]


                    
                    newStates_combine = np.zeros((length1 - 2, start1.shape[0], Nstate ))

                    
                        
                    #tableChunk1 = filledTable[start1[0], end2].reshape((1, Nstate))
                    #tableChunk2 = filledTable[start2, end1[0]].reshape((1, Nstate))
                    newValueVector1 = []
                    newValueVector2 = []
                    for valIndex in range(Nstate):
                        tableChunk = torch.zeros((1, Nstate)) - 5000
                        tableChunk[0, valIndex] = 0

                        tableChunk1 = filledTable[start1[choiceIndex], end2].reshape((1, Nstate))
                        tableChunk2 = filledTable[start2, end1[choiceIndex]].reshape((1, Nstate))

                        #print (tableChunk.shape, tableChunk2.shape)
                        valuesMini1 = model.doCombine(tableChunk, tableChunk2).data.numpy()
                        valuesMini1 = logsumexp(valuesMini1[0] + currentValuation )
                        newValueVector1.append(valuesMini1)

                        #tableChunk = torch.zeros((1, Nstate)) - 5000
                        #tableChunk[0, valIndex] = 0
                        valuesMini2 = model.doCombine(tableChunk1, tableChunk).data.numpy()
                        valuesMini2 = logsumexp(valuesMini2[0] + currentValuation )
                        newValueVector2.append(valuesMini2)
                    

                    newValueVector1 = np.array(newValueVector1)
                    newValueVector2 = np.array(newValueVector2)
                    valuations[nextQue, sampleNum] = newValueVector1
                    valuations[nextQue+1, sampleNum] = newValueVector2


    #printSeq = np.array([sequence.astype(str),structureString[:, 0] ]).T

    #print (list(structureString[:, 0]))
    #quit()
    #for sampleIndex in range(Nsample):
    #    print ('')
    #    #print (list(structureString[:, sampleIndex]))
    #    #quit()
    #    print ( ' '.join(list(sequence.astype(str)) ))
    #    print ( ' '.join(list(structureString[:, sampleIndex]) ))
    #    quit()

    #print (printSeq)
    #print (sequence)
    #print (structureString[:, 0])
    #quit()

    return structureString




def backtraceSubgraphNeural(sequence, Nstate, filledTable, model, cutOff = -10):
    

    #queList = np.zeros((sequence.shape[0]+10, Nsample, 2), dtype=int) - 1
    valuations = np.zeros((sequence.shape[0], sequence.shape[0], Nstate), dtype=float) - 5000
    valuations[0, -1] = 0

    usedGrid = np.zeros((sequence.shape[0], sequence.shape[0]), dtype=int)
    usedGrid[0, -1] = 1

    edgeStandard = np.zeros((sequence.shape[0], sequence.shape[0], 3), dtype=int)
    edgeCombine = np.zeros((sequence.shape[0], sequence.shape[0], sequence.shape[0]), dtype=int)
    
    sumValue = filledTable[0, -1].data.numpy()
    sumValue = logsumexp(sumValue)

    #print (logsumexp(filledTable.data.numpy(), axis=2))
    #quit()

    #cutOff = -10

    
    for length0 in range(0, sequence.shape[0] - 1): #Min lenght is 2
        length1 = sequence.shape[0] - length0

        start1 = np.arange(sequence.shape[0])
        end1 = start1 + length1 - 1
        argStay = np.argwhere(end1 < sequence.shape[0])[:, 0]
        start1 = start1[argStay]
        end1 = end1[argStay]


        
        #filled_now = filledTable[start1, end1].data.numpy()
        #values = logsumexp(filled_now + valuations_now, axis=1)

        #argValuable = np.argwhere(values > sumValue + cutOff)
        argValuable = np.argwhere(usedGrid[start1, end1] == 1)


        if argValuable.shape[0] > 0:
            argValuable = argValuable[:, 0]
            start1, end1 = start1[argValuable], end1[argValuable]

            valuations_now = valuations[start1, end1]

            
            #Add pair    
            tableChunk = filledTable[start1+1, end1-1]
            nuc1 = sequence[start1]
            nuc2 = sequence[end1]
            nucVector = np.zeros((start1.shape[0], 4, 4), dtype=int) - 5000
            nucVector[np.arange(start1.shape[0]), nuc1, nuc2] = 0
            nucVector = nucVector.reshape((start1.shape[0], 16))
            nucVector = torch.tensor(nucVector).float()
            newStates_pair = model.addPair(tableChunk, nucVector).data.numpy()
            
            
            #Add left
            nuc1 = sequence[start1]
            nucVector = np.zeros((start1.shape[0], 4), dtype=int) - 5000
            nucVector[np.arange(start1.shape[0]), nuc1] = 0
            nucVector = torch.tensor(nucVector).float()

            tableChunk = filledTable[start1+1, end1]
            newStates_left = model.addLeft(tableChunk, nucVector).data.numpy()
            #filledTable[start1, end1] = torch_logsumexp( filledTable[start1, end1] , newStates  )
            
            
            #Add right
            nuc2 = sequence[end1]
            nucVector = np.zeros((start1.shape[0], 4), dtype=int) - 5000
            nucVector[np.arange(start1.shape[0]), nuc2] = 0
            nucVector = torch.tensor(nucVector).float()

            tableChunk = filledTable[start1, end1-1]
            newStates_right = model.addRight(tableChunk, nucVector).data.numpy()
            #filledTable[start1, end1] = torch_logsumexp( filledTable[start1, end1] , newStates  )

            


            #option 4, combine two pieces
            if length1 >= 3:

                newStates_combine = np.zeros((length1 - 2, start1.shape[0], Nstate ))

                for adder1 in range(length1 - 2):
                    end2 = start1 + 1 + adder1
                    start2 = end2 + 1
                    tableChunk1 = filledTable[start1, end2]
                    tableChunk2 = filledTable[start2, end1]
                    newStates = model.doCombine(tableChunk1, tableChunk2).data.numpy()
                    newStates_combine[adder1] = np.copy(newStates)
                    #filledTable[start1, end1] = torch_logsumexp ( filledTable[start1, end1], newStates)

            else:

                newStates_combine = np.zeros((1, start1.shape[0], Nstate )) - 5000


            newStates_pair = newStates_pair.reshape((1, newStates_pair.shape[0], newStates_pair.shape[1]))
            newStates_left = newStates_left.reshape((1, newStates_left.shape[0], newStates_left.shape[1]))
            newStates_right = newStates_right.reshape((1, newStates_right.shape[0], newStates_right.shape[1]))
            newStates_cat = np.concatenate(( newStates_pair , newStates_left, newStates_right, newStates_combine  ), axis=0)

            #if start1[0] == 2:
            #    print ("B")
            #    print (newStates_pair[0, 0])
            #    print (newStates_left[0, 0])
            #    print (newStates_right[0, 0])

            valuations_reshape = valuations_now.reshape((1, valuations_now.shape[0], valuations_now.shape[1]))


            newStates_weighted = logsumexp(newStates_cat + valuations_reshape, axis=2)

            newStates_weighted_bool = np.zeros(newStates_weighted.shape, dtype=int)
            newStates_weighted_bool[newStates_weighted >= sumValue + cutOff ] = 1

            newStates_weighted_bool = newStates_weighted_bool.T

            newStates_weighted_bool_standard = newStates_weighted_bool[:, :3]
            newStates_weighted_bool_combine = newStates_weighted_bool[:, 3:]


            #argCheck = np.argwhere(np.logical_and(start1 == 14, end1==15))
            #if argCheck.shape[0] > 0:
            #    argCheck = argCheck[0][0]
            #    print ('length1', length1)
            #    print ('newStates_cat.shape', newStates_cat.shape)
            #    print (newStates_weighted[:, argCheck])
            #    print (sumValue + cutOff)
            #    print (newStates_cat[:, argCheck])
            #    print (valuations_reshape[:, argCheck])
            #    #print (filledTable[14+1, 15-1])
            #    quit()

            #print ("A")
            #print (start1.shape)
            #print ('newStates_cat', newStates_cat.shape)
            #print ('valuations_reshape', valuations_reshape.shape)
            #print (start1.shape)
            #print (edgeStandard[start1, end1].shape)
            #print (newStates_weighted_bool_standard.shape)
            edgeStandard[start1, end1] = newStates_weighted_bool_standard
            edgeCombine[start1, end1, :newStates_weighted_bool_combine.shape[1]] = newStates_weighted_bool_combine
            #print (newStates_weighted_bool_standard)

            #for a in range(start1.shape[0]):
            #    if np.sum(newStates_weighted_bool[a]) == 0:
            #        print ('dead: ', start1[a], end1[a])
            #        usedGrid[start1[a], end1[a]] = 2

            #print (start1, end1)
            #print (np.sum(newStates_weighted_bool))

            #Option 1: Add Pair
            argPair = np.argwhere(newStates_weighted_bool_standard[:, 0] == 1)
            if  argPair.shape[0] > 0:
                argPair = argPair[:, 0]
                nuc1 = sequence[start1[argPair]]
                nuc2 = sequence[end1[argPair]]
                nucVector = np.zeros((argPair.shape[0], 4, 4), dtype=int) - 5000
                nucVector[np.arange(argPair.shape[0]), nuc1, nuc2] = 0
                nucVector = nucVector.reshape((argPair.shape[0], 16))
                nucVector = torch.tensor(nucVector).float()

                newValueVector = np.zeros((argPair.shape[0], Nstate))
                for valIndex in range(Nstate):      
                    tableChunk = torch.zeros((argPair.shape[0], Nstate)) - 5000
                    tableChunk[:, valIndex] = 0
                    valuesMini = model.addPair(tableChunk, nucVector).data.numpy()
                    #print (valuesMini)
                    valuesMini = logsumexp(valuesMini + valuations_now[argPair] , axis=1)
                    newValueVector[:, valIndex] = valuesMini 

                valuations[start1[argPair]+1, end1[argPair]-1] = logsumexp(np.array([valuations[start1[argPair]+1, end1[argPair]-1], newValueVector]), axis=0)
                usedGrid[start1[argPair]+1, end1[argPair]-1] = 1

            
            
            argLeft = np.argwhere(newStates_weighted_bool_standard[:, 1] == 1)
            if argLeft.shape[0] > 0:
                #print ('hi2')
                #quit()

                argLeft = argLeft[:, 0]

                nuc1 = sequence[start1[argLeft]]
                nucVector = np.zeros((argLeft.shape[0], 4), dtype=int) - 5000
                nucVector[np.arange(argLeft.shape[0]), nuc1] = 0
                nucVector = torch.tensor(nucVector).float()

                

                newValueVector = np.zeros((argLeft.shape[0], Nstate))
                for valIndex in range(Nstate):      
                    tableChunk = torch.zeros((argLeft.shape[0], Nstate)) - 5000
                    tableChunk[:, valIndex] = 0
                    valuesMini = model.addLeft(tableChunk, nucVector).data.numpy()
                    valuesMini = logsumexp(valuesMini + valuations_now[argLeft], axis=1 )
                    
                    newValueVector[:, valIndex] = valuesMini

                
                valuations[start1[argLeft]+1, end1[argLeft]] = logsumexp(np.array([valuations[start1[argLeft]+1, end1[argLeft]], newValueVector]), axis=0)
                usedGrid[start1[argLeft]+1, end1[argLeft]] = 1
            

            argRight = np.argwhere(newStates_weighted_bool_standard[:, 2] == 1)
            if argRight.shape[0] > 0:
                argRight = argRight[:, 0]
                #print ('hi3')

                nuc2 = sequence[end1[argRight]]
                nucVector = np.zeros((argRight.shape[0], 4), dtype=int) - 5000
                nucVector[np.arange(argRight.shape[0]), nuc2] = 0
                nucVector = torch.tensor(nucVector).float()

                newValueVector = np.zeros((argRight.shape[0], Nstate))
                for valIndex in range(Nstate):      
                    tableChunk = torch.zeros((argRight.shape[0], Nstate)) - 5000
                    tableChunk[:, valIndex] = 0
                    valuesMini = model.addRight(tableChunk, nucVector).data.numpy()
                    #print ('valuations_now[argRight]', valuations_now[argRight])
                    valuesMini = logsumexp(valuesMini + valuations_now[argRight], axis=1 )
                    
                    newValueVector[:, valIndex] = valuesMini

                
                valuations[start1[argRight], end1[argRight]-1] = logsumexp(np.array([valuations[start1[argRight], end1[argRight]-1], newValueVector]), axis=0)
                usedGrid[start1[argRight], end1[argRight]-1] = 1
                #quit()

            
            argCombine0 = np.argwhere(np.sum(newStates_weighted_bool_combine, axis=1) >= 1)
            if argCombine0.shape[0] > 0:
                

                for adder1 in range(length1 - 2):
                    argCombine1 = np.argwhere(newStates_weighted_bool_combine[:, adder1] == 1)
                    if argCombine1.shape[0] > 0:
                        argCombine1 = argCombine1[:, 0]

                        start1_now = start1[argCombine1]
                        end1_now = end1[argCombine1]
                        end2 = start1_now + 1 + adder1
                        start2 = end2 + 1

                        newValueVector1 = np.zeros((argCombine1.shape[0], Nstate))
                        newValueVector2 = np.zeros((argCombine1.shape[0], Nstate))
                        for valIndex in range(Nstate):
                            tableChunk = torch.zeros((argCombine1.shape[0], Nstate)) - 5000
                            tableChunk[:, valIndex] = 0

                            tableChunk1 = filledTable[start1_now, end2]
                            tableChunk2 = filledTable[start2, end1_now]

                            #print (tableChunk.shape, tableChunk2.shape)
                            valuesMini1 = model.doCombine(tableChunk, tableChunk2).data.numpy()
                            valuesMini1 = logsumexp(valuesMini1 + valuations_now[argCombine1]  , axis=1)
                            #print (argCombine1.shape)
                            #print (valuesMini1.shape)
                            newValueVector1[:, valIndex]  = valuesMini1

                            argCheck = np.argwhere(np.logical_and(start2 == 14, end1_now==15))
                            

                            #tableChunk = torch.zeros((1, Nstate)) - 5000
                            #tableChunk[0, valIndex] = 0
                            valuesMini2 = model.doCombine(tableChunk1, tableChunk).data.numpy()
                            #if argCheck.shape[0] > 0:
                            #    argCheck = argCheck[0][0]
                            #   print ('m')
                            #    print (valuesMini2[argCheck])
                            
                            valuesMini2 = logsumexp(valuesMini2 + valuations_now[argCombine1] , axis=1)

                            
                            
                            newValueVector2[:, valIndex]  = valuesMini2

                        
                        valuations[start1_now, end2] = logsumexp(np.array([valuations[start1_now, end2], newValueVector1]), axis=0)
                        valuations[start2, end1_now] = logsumexp(np.array([valuations[start2, end1_now], newValueVector2]), axis=0)

                        #argCheck = np.argwhere(np.logical_and(start2 == 14, end1_now==15))
                        #if argCheck.shape[0] > 0:
                        #    argCheck = argCheck[0][0]
                        #    print ('hi')
                        #    #print (filledTable[9, 15])
                        #    #print (start1[argCheck], end1[argCheck])
                        #    print (newValueVector2[argCheck])
                        #    print (tableChunk1[argCheck])



                        usedGrid[start1_now, end2] = 1
                        usedGrid[start2, end1_now] = 1

    

    start1, end1 = np.arange(sequence.shape[0]), np.arange(sequence.shape[0])
    argStay = np.argwhere(usedGrid[start1, end1] >= 1)[:, 0]
    start1_now, end1_now = start1[argStay], end1[argStay]
    edgeStandard[start1_now, end1_now, 1] = 1
    edgeStandard[start1_now, end1_now, 2] = 1

    
    
    
    return edgeStandard, edgeCombine


def generateNeural(sequence, Nstate, structureMatrix, model, restriction=[], doSeq=True):

    #time1 = time.time()

    #doSeq = True



    filledTable = torch.zeros(( sequence.shape[0], sequence.shape[0], Nstate )).float() - 5000

    sequenceTable = torch.zeros(( sequence.shape[0], sequence.shape[0], Nstate )).float() - 5000

    if doSeq:
        structureMatrix_sum = np.sum(structureMatrix, axis=0)

    if len(restriction) > 0:
        edgeStandard = restriction[0]
        edgeCombine = restriction[1]
        usedGrid = np.sum(edgeStandard, axis=2) + np.sum(edgeCombine, axis=2)
        #restriction=[edgeStandard, edgeCombine]

    
    start1 = np.arange(sequence.shape[0]-1) + 1 
    end1 = start1 - 1
    filledTable[start1, end1, 0] = 0 #Initial states
    sequenceTable[start1, end1, 0] = 0
    

    
    
    timeSum1 = 0
    timeSum2 = 0

    for length1 in range(1, sequence.shape[0]+1):
        
        #print ("A", length1)

        

        start1 = np.arange(sequence.shape[0])
        end1 = start1 + length1 - 1
        argStay = np.argwhere(end1 < sequence.shape[0])[:, 0]
        start1 = start1[argStay]
        end1 = end1[argStay]

        if len(restriction) > 0:
            argStay = np.argwhere( usedGrid[start1, end1] >= 1 )[:, 0]
            start1 = start1[argStay]
            end1 = end1[argStay]

        if argStay.shape[0] > 0:
            
            time1 = time.time()

            argPair = np.arange(start1.shape[0])
            argLeft = np.arange(start1.shape[0])
            argRight = np.arange(start1.shape[0])
            if len(restriction) > 0:
                argPair = np.argwhere(edgeStandard[start1, end1, 0] > 0)[:, 0]
                argLeft = np.argwhere(edgeStandard[start1, end1, 1] > 0)[:, 0]
                argRight = np.argwhere(edgeStandard[start1, end1, 2] > 0)[:, 0]


            #option 1, add pair
            
            if length1 >= 2:
                if argPair.shape[0] > 0:

                    start1_now, end1_now = start1[argPair], end1[argPair]

                    tableChunk = filledTable[start1_now+1, end1_now-1]
                    nuc1 = sequence[start1_now]
                    nuc2 = sequence[end1_now]
                    nucVector = np.zeros((start1_now.shape[0], 4, 4), dtype=int) - 5000
                    nucVector[np.arange(start1_now.shape[0]), nuc1, nuc2] = 0
                    nucVector = nucVector.reshape((start1_now.shape[0], 16))
                    nucVector = torch.tensor(nucVector).float()

                    #print (nuc1, nuc2)
                    
                    newStates = model.addPair(tableChunk, nucVector)
                    #print (newStates)

                    filledTable[start1_now, end1_now] = torch_logsumexp(filledTable[start1_now, end1_now] , newStates)


                    if doSeq:
                        argSeq = np.argwhere(structureMatrix[start1_now, end1_now] == 1)[:, 0]
                        startSeq, endSeq = start1_now[argSeq], end1_now[argSeq]
                        tableChunkSeq = sequenceTable[startSeq+1, endSeq-1]
                        nucVectorSeq = nucVector[argSeq]
                        newStatesSeq = model.addPair(tableChunkSeq, nucVectorSeq)
                        sequenceTable[startSeq, endSeq] = torch_logsumexp(sequenceTable[startSeq, endSeq] , newStatesSeq)

                
                #print (filledTable.data.numpy())

            

                assert torch.max(sequenceTable) <= torch.max(filledTable)
            

            if argLeft.shape[0] > 0:

                start1_now, end1_now = start1[argLeft], end1[argLeft]

                nuc1 = sequence[start1_now]
                nucVector = np.zeros((start1_now.shape[0], 4), dtype=int) - 5000
                nucVector[np.arange(start1_now.shape[0]), nuc1] = 0
                nucVector = torch.tensor(nucVector).float()
                
                
                if length1 == 1:
                    tableChunk = torch.zeros((start1_now.shape[0], Nstate)) - 5000
                    tableChunk[:, 0] = 0

                if length1 >= 2:
                    tableChunk = filledTable[start1_now+1, end1_now]
                #print (tableChunk)
                newStates = model.addLeft(tableChunk, nucVector)
                filledTable[start1_now, end1_now] = torch_logsumexp( filledTable[start1_now, end1_now] , newStates  )

                if doSeq:
                    #print (start1)
                    argSeq = np.argwhere(structureMatrix_sum[start1_now] == 0)[:, 0]
                    startSeq, endSeq = start1_now[argSeq], end1_now[argSeq]
                    if length1 == 1:
                        tableChunkSeq = tableChunk[argSeq]
                    else:
                        tableChunkSeq = sequenceTable[startSeq+1, endSeq]
                    #print (tableChunkSeq)
                    nucVectorSeq = nucVector[argSeq]
                    newStatesSeq = model.addLeft(tableChunkSeq, nucVectorSeq)
                    sequenceTable[startSeq, endSeq] = torch_logsumexp(sequenceTable[startSeq, endSeq] , newStatesSeq)
        
                #print (sequenceTable[start1, end1])
                #print (filledTable[start1, end1])
                assert torch.max(sequenceTable) <= torch.max(filledTable)
            


            if argRight.shape[0] > 0:

                start1_now, end1_now = start1[argRight], end1[argRight]

                nuc2 = sequence[end1_now]
                nucVector = np.zeros((start1_now.shape[0], 4), dtype=int) - 5000
                nucVector[np.arange(start1_now.shape[0]), nuc2] = 0
                nucVector = torch.tensor(nucVector).float()
                
                if length1 == 1:
                    tableChunk = torch.zeros((start1_now.shape[0], Nstate)) - 5000
                    tableChunk[:, 0] = 0
                
                if length1 >= 2:
                    tableChunk = filledTable[start1_now, end1_now-1]
                newStates = model.addRight(tableChunk, nucVector)
                filledTable[start1_now, end1_now] = torch_logsumexp( filledTable[start1_now, end1_now] , newStates  )

                if doSeq:
                    argSeq = np.argwhere(structureMatrix_sum[end1_now] == 0)[:, 0]
                    startSeq, endSeq = start1_now[argSeq], end1_now[argSeq]
                    if length1 == 1:
                        tableChunkSeq = tableChunk[argSeq]
                    else:
                        tableChunkSeq = sequenceTable[startSeq, endSeq-1]
                    nucVectorSeq = nucVector[argSeq]
                    newStatesSeq = model.addRight(tableChunkSeq, nucVectorSeq)
                    sequenceTable[startSeq, endSeq] = torch_logsumexp(sequenceTable[startSeq, endSeq] , newStatesSeq)

                assert torch.max(sequenceTable) <= torch.max(filledTable)

            
            timeSum1 += (time.time() - time1)
            time2 = time.time()

            #option 4, combine two pieces
            if length1 >= 3:
                
                if len(restriction) == 0:

                    adder1 = np.arange(length1 - 2)

                    start1_now = np.copy(start1).reshape((-1, 1))
                    end1_now = np.copy(end1).reshape((-1, 1))
                    adder1 = adder1.reshape((1, -1))
                    zeros1 = np.zeros(adder1.shape, dtype=int)
                    start1_now = start1_now + zeros1
                    end1_now = end1_now + zeros1
                    
                    end2 = start1_now + 1 + adder1
                    start2 = end2 + 1

                    tableChunk1 = filledTable[start1_now, end2]
                    tableChunk2 = filledTable[start2, end1_now]

                    newStates = model.fastCombine(tableChunk1, tableChunk2)
                    filledTable[start1, end1] = torch_logsumexp ( filledTable[start1, end1], newStates)

                    if doSeq:
                        tableChunkSeq1 = sequenceTable[start1_now, end2]
                        tableChunkSeq2 = sequenceTable[start2, end1_now]
                        newStatesSeq = model.fastCombine(tableChunkSeq1, tableChunkSeq2)
                        sequenceTable[start1, end1] = torch_logsumexp(sequenceTable[start1, end1] , newStatesSeq)

                else:

                
                
                    for adder1 in range(length1 - 2):

                        argCombine0 = np.arange(start1.shape[0])
                        if len(restriction) > 0:
                            argCombine0 = np.argwhere(edgeCombine[start1, end1, adder1] > 0)[:, 0]

                        if argCombine0.shape[0] > 0:

                            start1_now, end1_now = start1[argCombine0], end1[argCombine0]

                            end2 = start1_now + 1 + adder1
                            start2 = end2 + 1
                            tableChunk1 = filledTable[start1_now, end2]
                            tableChunk2 = filledTable[start2, end1_now]
                            newStates = model.doCombine(tableChunk1, tableChunk2)
                            filledTable[start1_now, end1_now] = torch_logsumexp ( filledTable[start1_now, end1_now], newStates)

                            if doSeq:
                                tableChunkSeq1 = sequenceTable[start1_now, end2]
                                tableChunkSeq2 = sequenceTable[start2, end1_now]
                                newStatesSeq = model.doCombine(tableChunkSeq1, tableChunkSeq2)
                                sequenceTable[start1_now, end1_now] = torch_logsumexp(sequenceTable[start1_now, end1_now] , newStatesSeq)

                
            timeSum2 += (time.time() - time2)
            
            assert torch.max(sequenceTable) <= torch.max(filledTable)


    #print ("fill")
    #print ('')
    #print ('')
    #print ('')
    #print ('')
    #print (sequence.shape)
    #print (timeSum1, timeSum2)
    #quit()

    #fill1 = logsumexp( filledTable.data.numpy(), axis=2 )
    #fill1[fill1<0] = -1
    #fill1[fill1>100]  = fill1[fill1>100]  % 100
    #fill1[fill1>100] = 99
    #print (np.round(fill1).astype(int)  )

    #print (fill1[10, 15 ])

    #fill1 = logsumexp( sequenceTable.data.numpy(), axis=2 )
    #fill1[fill1<0] = -1
    #fill1[fill1>100] = 99
    #fill1[fill1>100]  = fill1[fill1>100]  % 100
    #print (np.round(fill1).astype(int)  )

    #print (fill1[10, 15 ])

    #print (np.argwhere( np.round(fill1).astype(int) == 12))

    
    if doSeq:
        return filledTable, sequenceTable
    else:
        return filledTable



def checkModel(sequence, Nstate, structureMatrix, model, restriction=[]):

    filledTable, sequenceTable = generateNeural(sequence, Nstate, structureMatrix, model, restriction=restriction)

    fullEnd = filledTable[0, -1]
    seqEnd = sequenceTable[0, -1]
    
    loss = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)

    loss = loss.data.numpy()
    
    return loss



def trainModel(sequences, Nstate, structures, Ntrain, saveModel, loadModel='', fullSubgrids=[]  ):

    
    iterNum = 2000 // sequences.shape[0]

    if loadModel == '':
        model = TransitionModel(Nstate)
    else:
        model = torch.load(loadModel)

    doSubgrids = False
    if len(fullSubgrids) != 0:
        doSubgrids = True
        fullSubgrids = fullSubgrids[0]
    
    

    #learningRate = 1e-2 #Slow

    learningRate = 5e-2 #Good
    
    #learningRate = 1e-1
    #learningRate = 3e-1
    #optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate, alpha=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, betas=(0.9, 0.99))

    
    lossAll = []
    iter = -1
    #for iter in range(iterNum):
    continue1 = True
    while continue1:
        iter += 1
        continue1 = False
        
        #if iter == 1:
        #    continue1 = False

        lossList = []

        for seqIndex in tqdm(range(Ntrain)):# range(sequences.shape[0]):

            #seqIndex = 22
            #print ('seqIndex', seqIndex)

            

            sequence = np.array(list(sequences[seqIndex]))
            sequence = checkConvertSequence(sequence)
            structure = list(structures[seqIndex])
            structureMatrix = structureToMatrix(structure)

            restriction = []
            if doSubgrids:
                edgeBoth = np.zeros((sequence.shape[0], sequence.shape[0], 3 + sequence.shape[0] ), dtype=int)

                subgridNow = fullSubgrids[fullSubgrids[:, 0] == seqIndex][:, 1:]

                #print ('subgrid size', subgridNow.shape)

                #print (edgeBoth.shape)
                #print (np.max(subgridNow, axis=0))

                edgeBoth[subgridNow[:, 0], subgridNow[:, 1], subgridNow[:, 2]] = 1
                edgeStandard = edgeBoth[:, :, :3]
                edgeCombine = edgeBoth[:, :, 3:]
                restriction = [edgeStandard, edgeCombine]


            

            filledTable, sequenceTable = generateNeural(sequence, Nstate, structureMatrix, model, restriction=restriction)

            

            fullEnd = filledTable[0, -1]
            seqEnd = sequenceTable[0, -1]

            loss = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)

            

            
            loss = loss * -1

            
            

            #if loss.data.numpy() < 500:
                
            lossList.append(loss.data.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            if seqIndex % 100 == 99:
                mean1 = np.mean(np.array(lossList)[-100:])
                median1 = np.median(np.array(lossList)[-100:])
                sort1 = np.sort(np.array(lossList)[-100:])
                #print (mean1, median1)
                stringRange = '[ ' +  str(sort1[25]) + ' ' + str(sort1[50]) + ' ' + str(sort1[75]) + ' ]'
                #print (mean1, stringRange)
                print ("Loss on the last 100 sequences: ", mean1)

                #plt.hist( np.array(lossList)[-100:] , bins=100 )
                #plt.show()

                torch.save(model, saveModel)


            if False:#seqIndex % 5000 == 4999:
        

                print ("Test: ")
                #quit()
                testLosses = []
                for seqIndex in range(Ntrain, sequences.shape[0]):#+20):
                    sequence = np.array(list(sequences[seqIndex]))
                    sequence = checkConvertSequence(sequence)
                    structure = list(structures[seqIndex])
                    structureMatrix = structureToMatrix(structure)


                    restriction = []
                    if doSubgrids:
                        edgeBoth = np.zeros((sequence.shape[0], sequence.shape[0], 3 + sequence.shape[0] ), dtype=int)

                        subgridNow = fullSubgrids[fullSubgrids[:, 0] == seqIndex][:, 1:]
                        edgeBoth[subgridNow[:, 0], subgridNow[:, 1], subgridNow[:, 2]] = 1
                        edgeStandard = edgeBoth[:, :, :3]
                        edgeCombine = edgeBoth[:, :, 3:]
                        restriction = [edgeStandard, edgeCombine]

                    loss = checkModel(sequence, Nstate, structureMatrix, model, restriction=restriction)

                    if loss > -500:
                        testLosses.append(loss)

                print (np.mean(np.array(testLosses)))

            

            #print (loss)
            
            #quit()


        #if iter % 10 == 0:
        #print (iter, iterNum)
        #print (loss)
        #print (np.mean(np.array(lossList)))

        lossMean = np.mean(np.array(lossList))


        #print (iter)
        #print (lossMean)

        if iter > 1:
            if lossMean > lossAll[-1] - 0.01:# +0.001:# 0.01:
                continue1 = False

        lossAll.append(lossMean)







        #print (torch.max(sequenceTable, axis=2)[0])

        #torch.save(model, saveModel)

        
    #quit()

    #print (fullEnd)
    #print (seqEnd)
    #print ('')
    #print (filledTable[0, -1])

    #print (np.max(filledTable.data.numpy(), axis=2))
    #print (np.max(sequenceTable.data.numpy(), axis=2))
    
    #print (torch.max(filledTable, axis=2)[0])
    #print (torch.max(sequenceTable, axis=2)[0])    

    return model
    




def sanityCheck():
    np.random.seed(1)

    pairMatrix = np.zeros((4, 4))
    pairMatrix[0, 2] = 1
    pairMatrix[2, 0] = 1
    pairMatrix[1, 3] = 1
    pairMatrix[3, 1] = 1


    length1 = 200
    #length1 = 20

    elementValues_initial = np.ones(5)

    sequences = makeSequences(1000, length1)

    for a in range(5, 1000):
        print (a)
        sequence = sequences[a]

        elementValues = np.ones(5)

        time1 = time.time()
        dynamicTable = parallelSolveDynamic(sequence, pairMatrix, elementValues)
        time2 = time.time()
        dynamicTable2 = confirmedParallelSolveDynamic(sequence, pairMatrix, elementValues)
        time3 = time.time()

        print (time2 - time1)
        print (time3 - time2)
        #quit()

        dynamicTable[dynamicTable < 0] = 0
        dynamicTable2[dynamicTable2<0] = 0

        #print (np.argwhere( np.abs(dynamicTable - dynamicTable2) != 0 ))


        assert np.mean(np.abs(dynamicTable - dynamicTable2)) == 0
        #print (np.mean(np.abs(dynamicTable - dynamicTable2)))
        print ('good')
    quit()


#sanityCheck()
#quit()



def filterRealSeq():

    #data = loadnpz('./data/realData/dbnFiles_short50.npz')
    data = loadnpz('./shortSequences.npz')
    sequences, structures = data[:, 0], data[:, 1]

    validChar = np.array(['A', 'C', 'G', 'U'])

    validList = np.zeros(sequences.shape[0], dtype=int)
    for a in range(sequences.shape[0]):
        seq1 = np.array(list(sequences[a]))
        seq1 = np.unique(seq1)

        if np.intersect1d(seq1, validChar).shape[0] != seq1.shape[0]:
            validList[a] = 1

    sequences, structures = sequences[validList == 0], structures[validList == 0]

    #print (structures.shape, sequences.shape)

    validParam = np.array(['(', '.', ')'])
    psuedoNots = np.zeros(sequences.shape[0], dtype=int)
    for a in range(sequences.shape[0]):
        struct1 = np.array(list(structures[a]))
        struct1 = np.unique(struct1)
        
        if np.intersect1d(struct1, validParam).shape[0] != struct1.shape[0]:
            psuedoNots[a] = 1

    #print (sequences.shape)
    sequences, structures = sequences[psuedoNots == 0], structures[psuedoNots == 0]


    emptyOnes = np.zeros(sequences.shape[0], dtype=int)

    for a in range(sequences.shape[0]):
        struct1 = np.array(list(structures[a]))
        struct1 = np.unique(struct1)
        if struct1.shape[0] == 1:
            emptyOnes[a] = 1

    
    sequences, structures = sequences[emptyOnes == 0], structures[emptyOnes == 0]

    #print (sequences.shape)
    #quit()
        

    
    return sequences, structures




def initialNeuralTesting():

    length1 = 5

    Nstate = 10
    #Nstate = 1

    


    sequence = np.array([0, 1, 1, 2, 3, 3, 0, 3, 2, 1], dtype=int)
    

    pairMatrix = np.zeros((4, 4))
    pairMatrix[0, 2] = 1
    pairMatrix[2, 0] = 1
    pairMatrix[1, 3] = 1
    pairMatrix[3, 1] = 1
    
    #test8
    #NumSeq = 50
    NumSeq = 2
    length1 = 3

    np.random.seed(0)

    elementValues_initial = np.ones(5)
    #elementValues_True = np.random.random(size=5)

    sequences = makeSequences(NumSeq, length1)

    sequences = sequences[1:]
    NumSeq = 1

    #print (sequences[0])
    #quit()
    
    
    #structure, countVector = giveStructure(sequences[0], pairMatrix, elementValues_initial)
    #structureMatrix = structureToMatrix(structure)

    sequenceMatrices = np.zeros((NumSeq, length1, length1), dtype=int)
    for a in range(NumSeq):
        structure, countVector = giveStructure(sequences[a], pairMatrix, elementValues_initial)
        structureMatrix = structureToMatrix(structure)
        sequenceMatrices[a] = np.copy(structureMatrix)

    #structureMatrix = structureMatrix.reshape((1, structureMatrix.shape[0], structureMatrix.shape[1]))



    #filledTable, sequenceTable = generateNeural(sequence, Nstate, structureMatrix)
    sequence = sequences[0]

    model = trainModel(sequences, Nstate, sequenceMatrices)
    quit()


    sequence = np.array([0, 1, 1, 3, 2, 3, 0, 3, 2, 1], dtype=int)
    #simpleSequencePair = np.array([0, 1, 2, 2, 0, 1, 0, 3, 0, 3])
    simpleSequencePair = np.array([0, 1, 2, 2, 0, 1, 0, 0, 3, 3])

    structureMatrix = simpleToSequenceMatrix(simpleSequencePair)

    print ('')
    print ('on new seq')
    loss = checkModel(sequence, Nstate, structureMatrix, model)
    print (loss)
    #print (np.max(filledTable.data.numpy(), axis=2))
    #print (np.max(sequenceTable.data.numpy(), axis=2))



#initialNeuralTesting()
#quit()





def testNeural():

    pairMatrix = np.zeros((4, 4))
    pairMatrix[0, 2] = 1
    pairMatrix[2, 0] = 1
    pairMatrix[1, 3] = 1
    pairMatrix[3, 1] = 1
    
    #model 5
    #NumSeq = 600
    #length1 = 20
    #Ntrain = 500

    #test8
    NumSeq = 250
    #length1 = 20
    length1 = 40
    Ntrain = 200

    np.random.seed(0)

    #elementValues_initial = np.ones(5)
    #elementValues_initial = np.array([-1, 3, -1, -1, -1]) #Only stacking gets rewarded
    #elementValues_initial = np.array([-0.1, -0.1, -0.1, -0.1, 3])
    elementValues_initial = np.array([0.417, 0.720, 0.0001, 0.302, 0.146])
    #elementValues_True = np.random.random(size=5)

    #sequences = makeSequences(NumSeq, length1)

    sequenceMatrices = np.zeros((NumSeq, length1, length1), dtype=int)
    for a in range(NumSeq):
        #print ('')
        #print (' '.join(list(sequences[a].astype(str))))
        structure, countVector = giveStructure(sequences[a], pairMatrix, elementValues_initial)
        #print (' '.join(structure))
        structureMatrix = structureToMatrix(structure)
        sequenceMatrices[a] = np.copy(structureMatrix)

        
    

    #quit()

    modelName = './data/neural/model/6.pt'

    Nstate = 10
    #Nstate = 2
    model = trainModel(sequences, Nstate, sequenceMatrices, Ntrain, modelName)

    #torch.save(model, './data/neural/model/4.pt')
    #quit()
    #model = torch.load()




    print ("Test: ")
    testLosses = []
    for a in range(Ntrain, NumSeq):
        loss = checkModel(sequences[a], Nstate, sequenceMatrices[a], model)
        testLosses.append(loss)

    print (np.mean(np.array(testLosses)))

    
    quit()



#testNeural()#
#quit()


def calculateAllSubgrids(sequences, Nstate, model):

    #fullInfo = []

    fullInfo = np.zeros((0, 4), dtype=int)


    for seqIndex in range(sequences.shape[0]):

        print (seqIndex)

        sequence = np.array(list(sequences[seqIndex]))
        print (len(sequence))
        sequence = checkConvertSequence(sequence)

        structureMatrix = ''
        filledTable = generateNeural(sequence, Nstate, structureMatrix, model, restriction=[], doSeq=False)

        edgeStandard, edgeCombine = backtraceSubgraphNeural(sequence, Nstate, filledTable, model, cutOff = -10)

        edgeConcat = np.concatenate((edgeStandard, edgeCombine), axis=2)
        edgeConcat_storage = np.argwhere(edgeConcat == 1)

        print (np.max(edgeConcat_storage, axis=0))

        index1 = np.zeros(edgeConcat_storage.shape[0], dtype=int).reshape((-1, 1)) + seqIndex
        edgeConcat_storage = np.concatenate(( index1, edgeConcat_storage ), axis=1)


        fullInfo = np.concatenate((fullInfo, edgeConcat_storage), axis=0)

        #fullInfo.append(np.copy(edgeConcat_storage))

    return fullInfo

    





def realTrainNeural(saveModel):

    
    sequences, structures = filterRealSeq()

    #print (sequences.shape)


    
    np.random.seed(0)
    perm1 = np.random.permutation(sequences.shape[0])
    sequences, structures = sequences[perm1], structures[perm1]
    


    Ntrain = sequences.shape[0] - 100

    #Nstate = 10


    
    #loadModel = './data/neural/model/9.pt'
    #model = torch.load(loadModel)


    #fullSubgrids = calculateAllSubgrids(sequences, Nstate, model)
    #np.savez_compressed('./data/neural/subgrid/model7.npz', fullSubgrids)
    #quit()

    #fullSubgrids = loadnpz('./data/neural/subgrid/model7.npz')
    #sequences = sequences[:2]
    #structures = structures[:2]
    #Ntrain = 2
    #print ('')
    #print ('')
    #for a in range(2):
    #    fullSubgrids1 = fullSubgrids[fullSubgrids[:, 0] == a]
    #    print (np.max(fullSubgrids1, axis=0))
    #    print (len(sequences[a]))
    #quit()

    #saveModel = '.15.pt'

    #saveModel = './' + saveModel
    #loadModel =  './data/neural/model/14.pt'
    Nstate = 20
    #Nstate = 5
    model = trainModel(sequences, Nstate, structures, Ntrain, saveModel)#, loadModel=loadModel)#, loadModel=loadModel)


    quit()

    

    Nstate = 10
    #model = trainModel(sequences, Nstate, structures, Ntrain, saveModel, loadModel=loadModel, fullSubgrids=[fullSubgrids])
    #model = trainModel(sequences, Nstate, structures, Ntrain, saveModel, loadModel=loadModel)

    #torch.save(model, './data/neural/model/4.pt')
    #quit()
    #model = torch.load()




    print ("Test: ")
    testLosses = []
    for a in range(Ntrain, NumSeq):
        loss = checkModel(sequences[a], Nstate, sequenceMatrices[a], model)
        testLosses.append(loss)

    print (np.mean(np.array(testLosses)))



#realTrainNeural()
#quit()






def realNeuralBacktrace():


    model = torch.load('./data/neural/model/7.pt')
    sequences, structures = filterRealSeq()

    #a = 8

    for a in range(18, 20):

        sequence = sequences[a]
        structure = structures[a]


        sequence = np.array(list(sequences[a]))
        print (' '.join(list(sequence)))
        sequence = checkConvertSequence(sequence)
        structureMatrix = structureToMatrix(list(structure))

        

        print (' '.join(list(structure)))
        Nstate = 10
        filledTable, sequenceTable = generateNeural(sequence, Nstate, structureMatrix, model)

        fullEnd = filledTable[0, -1]
        seqEnd = sequenceTable[0, -1]
        loss_true = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)
        print ('loss_true', loss_true)

        edgeStandard, edgeCombine = backtraceSubgraphNeural(sequence, Nstate, filledTable, model)

        #. . . . . . ( ( ( ( ( ( . . . . . ) ) ) ) ) ) . . . . . . . . . . . ( ( ( ( . . . ( ( ( ( . . . . . . . . . . . ) ) ) ) . . ( ( ( ( . . . . . . . . . . . ) ) ) ) . . ) ) ) ) . . . . . . . . . . . ( ( ( ( ( ( ( . . . . ) ) ) ) ) ) ) . . . . . . . . . . ( ( ( ( ( ( . . . . . ) ) ) ) ) ) . . . . . . . . . . . . . . . . . . . . . .
        #. . . . . . ( ( ( ( ( ( . . . . . ) ) ) ) ) ) . . . . . . . . . . . ( ( ( ( . . . ( ( ( ( . . . . . . . . . . . ) ) ) ) . . ( ( ( ( . . . . . . . . . . . ) ) ) ) . . ) ) ) ) . . . . . . . . . . . ( ( ( ( ( ( ( . . . . ) ) ) ) ) ) ) . . . . . . . . . . ( ( ( ( ( ( . . . . . ) ) ) ) ) ) . . . . . . . . . . . . . . . . . . . . . .


        for b in range(10):
            
            Nsample = 1
            structureString = backtraceNeural(sequence, Nstate, Nsample, filledTable, model)
            structureString = structureString[:, 0]
            structureOurs = list(structureString)

            print (' '.join(structureOurs))

            structureMatrix_our = structureToMatrix(list(structureOurs))


            filledTable_new, sequenceTable_new = generateNeural(sequence, Nstate, structureMatrix_our, model, restriction=(edgeStandard, edgeCombine))
            fullEnd = filledTable_new[0, -1]
            seqEnd = sequenceTable_new[0, -1]
            loss = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)
            print (loss)

        #print (' '.join(structureOurs))

#realNeuralBacktrace()
#quit()



def realAccuracy():

    

    #Interesting example (9): 
    #[2 2 0 1 3 1 2 2 1 3 3 2 1 3 2 0 2 2 1 2 1 0 1 2 2 1 0 0 2 0 2 2 1 2 0 2 2 0 2]
    #. . . ( ( ( ( . ( ( ( ( ( ( ( . . . . . . . ) ) ) ) ) ) ) . . . ) ) ) . ) . .
    #loss_true -8.876335
    #. . . ( ( ( ( . ( ( ( ( ( ( ( . . . . . . . ) ) ) ) ) ) ) . . . ) ) ) ) . . .
    #lossBest -0.5005646

    #test set 50 accuracy: [0.48 0.68 0.8 ]



    #errorAll = loadnpz('./data/neural/score/errors_11.npz')

    #print (np.mean(errorAll[:50], axis=0))

    #quit()

    #Nstate = 10
    Nstate = 20
    

    model = torch.load('./data/neural/model/14.pt')
    sequences, structures = filterRealSeq()

    np.random.seed(0)
    perm1 = np.random.permutation(sequences.shape[0])
    sequences, structures = sequences[perm1], structures[perm1]

    sequences, structures = sequences[-100:], structures[-100:]

    #Nsample = 10
    Nsample = 100

    errorAll = np.zeros((100, 4))

    interesting = []

    for a in range(100):# range(0, sequences.shape[0]): #[5, 7]:#

        print ('')
        print (a)

        sequence = sequences[a]
        structure = structures[a]

        #. ( ( ( ( ( ( ( ( ( ( . . ( ( ( ( ( ( ( ( ( ( ( . . . . . . . . . . . . . . . . ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) .
        #( ( ( ( ( ( ( ( ( ( ( . . ( ( ( ( ( ( ( ( ( ( ( . . . . . . . . . . . . . . . . ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )
        #UGUAAACAUCCUUGACUGGAAGCUGUAAGGUGUUCAGAGGAGCUUUCAGUCGGAUGUUUACA



        #( ( ( ( ( ( ( . . ( ( ( ( . . . . . . . . . ) ) ) ) ( ( ( ( ( ( . . . . . . . ) ) ) ) ) ) . . . . ( ( ( ( ( . . . . . . . ) ) ) ) ) ) ) ) ) ) ) ) . . . .
        #( ( ( ( ( ( ( . . ( ( ( ( . . . . . . . . . ) ) ) ) . ( ( ( ( ( ( ( . . . ) ) ) ) ) ) ) . . . . . ( ( ( ( ( . . . . . . . ) ) ) ) ) ) ) ) ) ) ) ) . . . .


        #. ( ( ( ( ( . . . . . . . . . . . . . . . . . . . . ) ) ) ) ) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        #( ( ( ( ( ( . . . ( ( ( ( ( . . . . ) ) ) ) ) . . . ) ) ) ) ) ) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


        #. . . . . . . . ( ( ( . ( ( . . . ( ( . . . ( ( ( ( . . ( ( ( ( ( ( ( ( . . . . . ) ) ) . . ) ) ) ) ) . ) ) ) ) . . . ) ) . . . ) ) . ) ) ) . . . . . . . .
        #( ( ( ( ( . . . . . . . ) ) ) ) ) . ( ( ( ( ( ( ( ( . ( ( ( ( ( ( ( ( ( ( . . . . . . . . . . . . . ) ) ) ) ) ) ) ) ) ) . ) ) ) ) . ) ) ) ) . . . . . . . .

        print ('seq')
        print (sequence)


        sequence = np.array(list(sequences[a]))
        #print (' '.join(list(sequence)))
        sequence = checkConvertSequence(sequence)
        structureMatrix = structureToMatrix(list(structure))

        
        print (sequence)
        print (' '.join(list(structure)))
        
        filledTable, sequenceTable = generateNeural(sequence, Nstate, structureMatrix, model)

        fullEnd = filledTable[0, -1]
        seqEnd = sequenceTable[0, -1]
        loss_true = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)
        loss_true = loss_true.data.numpy()
        #print ('loss_true', loss_true)

        edgeStandard, edgeCombine = backtraceSubgraphNeural(sequence, Nstate, filledTable, model, cutOff=-20)

        #. . . . . . ( ( ( ( ( ( . . . . . ) ) ) ) ) ) . . . . . . . . . . . ( ( ( ( . . . ( ( ( ( . . . . . . . . . . . ) ) ) ) . . ( ( ( ( . . . . . . . . . . . ) ) ) ) . . ) ) ) ) . . . . . . . . . . . ( ( ( ( ( ( ( . . . . ) ) ) ) ) ) ) . . . . . . . . . . ( ( ( ( ( ( . . . . . ) ) ) ) ) ) . . . . . . . . . . . . . . . . . . . . . .
        #. . . . . . ( ( ( ( ( ( . . . . . ) ) ) ) ) ) . . . . . . . . . . . ( ( ( ( . . . ( ( ( ( . . . . . . . . . . . ) ) ) ) . . ( ( ( ( . . . . . . . . . . . ) ) ) ) . . ) ) ) ) . . . . . . . . . . . ( ( ( ( ( ( ( . . . . ) ) ) ) ) ) ) . . . . . . . . . . ( ( ( ( ( ( . . . . . ) ) ) ) ) ) . . . . . . . . . . . . . . . . . . . . . .

        print ('loss_true', loss_true)

        #Nsample2 = 1000
        #Nsample2 = 100
        Nsample2 = 10
        structureString = backtraceNeural(sequence, Nstate, Nsample2, filledTable, model)



        ourStructureList = []
        for b in range(Nsample2):    
            structureOurs = structureString[:, b]
            structureOurs = ''.join(list(structureOurs))
            ourStructureList.append(structureOurs)

        ourStructureList = np.array(ourStructureList)

        ourStructureList, counts1 = np.unique(ourStructureList, return_counts=True)
        ourStructureList = ourStructureList[np.argsort(counts1)[-1::-1]]
        ourStructureList = ourStructureList[:100]
        
        
        structureBest = ''
        lossBest = -5000
        lossList = []
        structureList = []

        for structureOurs in ourStructureList:

            structureOurs_string = structureOurs

            structureOurs = list(structureOurs)

            #print (' '.join(structureOurs))

            #print (structureOurs)

            structureMatrix_our = structureToMatrix(structureOurs)


            filledTable_new, sequenceTable_new = generateNeural(sequence, Nstate, structureMatrix_our, model, restriction=(edgeStandard, edgeCombine))
            fullEnd = filledTable_new[0, -1]
            seqEnd = sequenceTable_new[0, -1]
            loss = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)
            #print (loss)

            loss = loss.data.numpy()

            lossList.append(loss)
            structureList.append(structureOurs_string)

            if loss > lossBest:
                lossBest = loss
                structureBest = structureOurs
                


        
        structureBest = list(structureBest)
        print (' '.join(structureBest))

        print ('lossBest', lossBest)

        #print (lossList)


        error1, sum1 = evaluateAccuracy(list(structure), structureBest)

        print ('error, sum:', error1, sum1)

        #allErrors.append([error1, sum1])


        #Top 10:
        lossList = np.array(lossList)
        structureList = np.array(structureList)
        print (structureList.shape)
        print (lossList.shape)
        topArg = np.argsort(lossList * -1)[:10]
        errorList = []
        for b in range(topArg.shape[0]):
            structureTop = structureList[topArg[b]]
            error2, sum2 = evaluateAccuracy(list(structure), list(structureTop))
            errorList.append(error2)
        errorList = np.array(errorList)
        errorMin = np.min(errorList)

        errorAll[a, 0] = sum1
        errorAll[a, 1] = errorList[0]
        errorAll[a, 2] = np.min(errorList[:2])
        errorAll[a, 3] = np.min(errorList[:10])

        accuracy1 = np.copy(errorAll[:a+1])
        accuracy1[accuracy1!=0] = -1
        accuracy1 = accuracy1 + 1
        print ("accuracy")
        print (np.mean(accuracy1[:, 1:], axis=0))
        print ('errors')
        print (np.mean(errorAll[:a+1, 1:], axis=0))



        

        if errorMin < error1:
            print ("Error Interesting")
            interesting.append(a)
            print (interesting)




        #errorList = np.array(errorList)

        if lossBest < loss_true:
            for b in range(10):
                print ('issue')
            print ("Issue, did not print best!!")

        #quit()
        np.savez_compressed('./data/neural/score/errors_11.npz', errorAll)

        #quit()

#realAccuracy()
#quit()

def interpretStates():


    Nstate = 20
    

    model = torch.load('./data/neural/model/14.pt')
    sequences, structures = filterRealSeq()

    np.random.seed(0)
    perm1 = np.random.permutation(sequences.shape[0])
    #sequences, structures = sequences[perm1], structures[perm1]
    #sequences, structures = sequences[-100:], structures[-100:]

    
    stackStruct = [ structures[0][12:21], structures[0][11:22], structures[0][10:23], structures[0][9:24] ]
    stackSeq = [ sequences[0][12:21], sequences[0][11:22], sequences[0][10:23], sequences[0][9:24] ]
    
    
    print (structures[0])
    quit()

    for a in range(len(stackStruct)):
        sequence = stackSeq[a]
        structure = stackStruct[a]

        sequence = np.array(list(sequence))
        sequence = checkConvertSequence(sequence)

        print (structure)
        
        structureMatrix = structureToMatrix(list(structure))

        filledTable, sequenceTable = generateNeural(sequence, Nstate, structureMatrix, model)

        vector1 = sequenceTable[0, -1].data.numpy()

        if a == 0:
            vector0 = vector1

        else:
            plt.plot(vector1 - vector0)
    plt.show()

    

#interpretStates()
#quit()





def ViennaAccuracy():

    Nstate = 20

    #The command is "RNAfold". 

    #import viennarna
    #quit()

    errorAll = loadnpz('./data/neural/score/errors_11.npz')
    errorOurs = errorAll[:, 1]
    #print (errorAll.shape)
    #print (errorAll[:10])
    #quit()
    
    sequences, structures = filterRealSeq()

    np.random.seed(0)
    perm1 = np.random.permutation(sequences.shape[0])
    sequences, structures = sequences[perm1], structures[perm1]
    sequences, structures = sequences[-100:], structures[-100:]

    file1 = open('./data/realData/100_structures', 'r')
    structuresVienna0 = file1.readlines()
    structuresVienna = []
    for a in range(len(structuresVienna0)):
        if a % 3 == 2:
            structure = structuresVienna0[a]
            structure = structure.split(' ')[0]
            structuresVienna.append(structure)

    errorList = np.zeros(100)
    perfectList = np.zeros(100)
    lengths = np.zeros(100)

    errorListVienna = []
    
    for a in range(len(structuresVienna)):

        structureVienna = structuresVienna[a]
        structureTrue = structures[a]

        lengths[a] = len(structureTrue)
        
        error2, sum2 = evaluateAccuracy(list(structureTrue), list(structureVienna))

        errorList[a] = error2

        if error2 == 0:
            perfectList[a] = 1

        errorListVienna.append(error2)


        print (error2, sum2)

    errorListVienna = np.array(errorListVienna)


    max1 = max(np.max(errorListVienna), np.max(errorOurs) )

    plt.hist(errorOurs, range=(0, max1), alpha=0.5)
    plt.hist(errorListVienna, range=(0, max1), alpha=0.5)
    plt.legend(['my algorithm', 'ViennaFold'])
    plt.xlabel("error")
    plt.ylabel('count')
    plt.show()

    #print (np.mean(lengths))
    quit()

    print (np.mean(errorList))
    print (np.mean(perfectList))

    quit()

    

    strings1 = []
    for a in range(100):
        string1 = '>seq' + str(a) + '\n'
        string2 = sequences[a] + '\n'
        strings1.append(string1)
        strings1.append(string2)
    

    with open('./data/realData/100.fasta', 'a') as the_file:
        for string1 in strings1:
            print (string1)
            the_file.write(string1)
    
    quit()


    structPred = []
    structPred.append('(((((((......(((.((((((((..(((((((........)))).)))..))))..)))))))))))))).')
    structPred.append('.((.((((((((.((((((((((.(((..((((((((.......))))))))..))).)))))))))).)))))))).))')
    structPred.append('..(((....)))(((((((.((...(((((.......)))))...))))))))).....')
    structPred.append('(((((((....((((((((.........))))))))....(((....)))((((.......)))).))))))).')
    structPred.append('(((((((..((((.........)))).(((((.......))))).....(((((.......))))))))))))....')

    #(((((((......(((.((((((((..(((((((........)))).)))..))))..)))))))))))))).

    #0.8 from Mine. 0.0 from Vienna. First 5. 

    for a in range(5):
        
        print (sequences[a])
        print (structures[a])
        print (structPred[a])
        #quit()

        if structures[a] == structPred[a]:
            print ("Good")
        else:
            print ("Bad")

        print (sequences[a])

#ViennaAccuracy()
#quit()




def testNeuralBacktrace():

    model = torch.load('./data/neural/model/5.pt')
    #model = torch.load('./data/neural/model/6.pt')

    pairMatrix = np.zeros((4, 4))
    pairMatrix[0, 2] = 1
    pairMatrix[2, 0] = 1
    pairMatrix[1, 3] = 1
    pairMatrix[3, 1] = 1

    #1 in 10,000 error. Awesome
    

    NumSeq = 10
    length1 = 20
    #length1 = 60
    #length1 = 100 #Issue, seed 6
    #length1 = 200

    #rand1 = np.random.randint(1000)
    for rand1 in range(100, 1000):#range(61, 1000):
    
        #rand1 = 6
        print ('seed', rand1)
        np.random.seed(rand1)

        #elementValues_initial = np.array([-1, 3, -1, -1, -1])
        #elementValues_initial = np.array([-0.1, -0.1, -0.1, -0.1, 3])
        #elementValues_True = np.random.random(size=5)
        elementValues_initial = np.array([0.417, 0.720, 0.0001, 0.302, 0.146])
        #print (elementValues_True)
        #quit()

        el1 = np.array([21, 46, 1, 3, 9])
        el2 = np.array([23, 45, 1, 3, 8])

        #print (np.sum(elementValues_initial * el1))
        #print (np.sum(elementValues_initial * el2))
        #quit()

        sequences = makeSequences(NumSeq, length1)
        sequence = sequences[2]

        #sequenceMatrices = np.zeros((NumSeq, length1, length1), dtype=int)
        #for a in range(NumSeq):
        structure, countVector = giveStructure(sequence, pairMatrix, elementValues_initial)

        elementCount = countStructure(structure)

        print (' '.join(structure))
        

        sequence_str = ' '.join(list(sequence.astype(str)))
        structure_str = ' '.join(structure)
        #print (sequence_str)
        #print (structure_str)
        #quit()
        #print (structure)
        #quit()
        structureMatrix = structureToMatrix(structure)
        #sequenceMatrices[a] = np.copy(structureMatrix)
        Nstate = 10

        

        
        time1 = time.time()
        filledTable, sequenceTable = generateNeural(sequence, Nstate, structureMatrix, model)
        timelength1 = time.time() - time1

        fullEnd = filledTable[0, -1]
        seqEnd = sequenceTable[0, -1]
        loss_dynamic = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)

        probFull1 = torch.logsumexp(fullEnd, dim=0).data.numpy()

        print (torch.logsumexp(seqEnd, dim=0), torch.logsumexp(fullEnd, dim=0))
        
        #quit()
        
        edgeStandard, edgeCombine = backtraceSubgraphNeural(sequence, Nstate, filledTable, model)#, cutOff=-20)


        usedGrid = np.sum(edgeStandard, axis=2) + np.sum(edgeCombine, axis=2)
        usedGrid[usedGrid!=0] = 1

        #edgeStandard[np.arange(length1), np.arange(length1), 1] = 1
        #edgeStandard[np.arange(length1), np.arange(length1), 2] = 1

        #usedGrid[np.arange(length1), np.arange(length1)] = -1 * usedGrid[np.arange(length1), np.arange(length1)]
        #usedGrid[np.arange(length1), np.arange(length1)] = usedGrid[np.arange(length1), np.arange(length1)] - 1

        #print ('usedGrid')
        #print (edgeStandard[:, :, 0])
        #print (usedGrid)

        
        #print (edgeCombine[10, 15])

        #edgeCombine[10, 15, :] = 1
        #edgeStandard[10, 15, :] = 1
        #quit()

        #print (10 13)
        #print (edgeStandard[14, 15])
        #print (edgeCombine[14, 15])
        #quit()
        #quit()
        #quit()
        time1 = time.time()
        filledTable, sequenceTable = generateNeural(sequence, Nstate, structureMatrix, model, restriction=[edgeStandard, edgeCombine])
        timelength2 = time.time() - time1

        fullEnd = filledTable[0, -1]
        seqEnd = sequenceTable[0, -1]
        loss_dynamic = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)

        probFull2 = torch.logsumexp(fullEnd, dim=0).data.numpy()

        print (torch.logsumexp(seqEnd, dim=0), torch.logsumexp(fullEnd, dim=0))

        if probFull1 - probFull2 > 0.5:
            print ("inaccurate")
            quit()

        val1 = torch.logsumexp(seqEnd, dim=0).data.numpy()
        if val1 < 0:
            print ("ISSUE negative")
            quit()

        #quit()


        #print ('times', timelength1, timelength2, timelength1/timelength2  )

    

    quit()


    structure_list = []
    loss_list = []
    
    print (elementCount)

    for a in range(0, 30):
        np.random.seed(a)

        print (a, 30)

        
        Nsample = 1
        structureString = backtraceNeural(sequence, Nstate, Nsample, filledTable, model)
        structureString = structureString[:, 0]
        structureOurs = list(structureString)

        structure_list.append(copy.deepcopy(structureOurs))

        #print (sequence_str)
        #print (' '.join(structureOurs))

        elementOurs = countStructure(structureOurs)
        

        #print (a)
        #print (elementOurs)
        #quit()

        #print ('losses')
        #print (loss_dynamic)

        

        
        
        #structureBad = ['(', '(', '.', '.', '.', '.', '.', '.', '.', '.', ')', ')', '.', '.', '.', '.', '.', '.', '.', '.']
        #structureBad = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
        #structureBad = ['(', '.', '(', '.', '(', '.', '.', '.', '.', '.', '.', ')', '.', '.', '.', '.', ')', '.', '.', ')']


        elementValues_bad = np.array([1, 3, 1, 1, 1])
        structureBad, countVector = giveStructure(sequence, pairMatrix, elementValues_bad)

        structureMatrixBad = structureToMatrix(structureOurs)

        filledTable_new, sequenceTable_new = generateNeural(sequence, Nstate, structureMatrixBad, model)

        fullEnd = filledTable_new[0, -1]
        seqEnd = sequenceTable_new[0, -1]
        loss = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)
        print (loss)
        print (elementOurs)

        print (np.sum(elementValues_initial * elementOurs))
        #quit()

        loss_list.append(loss.data.numpy())


    #print (elementCount)
    
    loss_list = np.array(loss_list)
    #print(loss_list)

    argmin = np.argmax(loss_list)
    structure_best = structure_list[argmin]

    elementOurs = countStructure(structure_best)

    #print (elementOurs)

    #print (structure)


    #print (filledTable.shape)

    
    
#testNeuralBacktrace()
#quit()






def plotDynamicResult():



    #test4
    #NumSeq = 10
    #length1 = 400
    #Ntrain = 1

    #test5
    #NumSeq = 10
    #length1 = 900
    #Ntrain = 1

    #test6
    #NumSeq = 10
    #length1 = 1500
    #Ntrain = 1

    #test7
    #NumSeq = 10
    #length1 = 2400
    #Ntrain = 1


    testFolders = ['./data/tests/test4/', './data/tests/test5/', './data/tests/test6/', './data/tests/test7/']
    #testFolders = ['./data/tests/test4/', './data/tests/test8/']

    vals = []

    for testFolder in testFolders:

        valueListFull = loadnpz(testFolder + 'valueList.npz')
        valueListFull = np.mean(valueListFull[:, :, 1:], axis=2)
        print (np.mean(valueListFull, axis=0))
        vals.append(np.mean(valueListFull, axis=0))

    vals = np.array(vals)
    plt.plot(vals[:, 2], vals[:, 0])
    plt.plot(vals[:, 2], vals[:, 1])
    plt.plot(vals[:, 2], vals[:, 2])
    plt.scatter(vals[:, 2], vals[:, 0])
    plt.scatter(vals[:, 2], vals[:, 1])
    plt.show()






def givePrediction(sequence, modelname):

    Nstate = 20
    

    model = torch.load(modelname)
    
    Nsample = 100

    errorAll = np.zeros((100, 4))

    interesting = []

    


    sequence = np.array(list(sequence))
    sequence = checkConvertSequence(sequence)
    
    seqMatrix = np.zeros((len(sequence), len(sequence)), dtype=int)
    
    filledTable, sequenceTable = generateNeural(sequence, Nstate, seqMatrix, model)

    
    edgeStandard, edgeCombine = backtraceSubgraphNeural(sequence, Nstate, filledTable, model, cutOff=-20)

    

    #Nsample2 = 1000
    #Nsample2 = 100
    Nsample2 = 10
    structureString = backtraceNeural(sequence, Nstate, Nsample2, filledTable, model)



    ourStructureList = []
    for b in range(Nsample2):    
        structureOurs = structureString[:, b]
        structureOurs = ''.join(list(structureOurs))
        ourStructureList.append(structureOurs)

    ourStructureList = np.array(ourStructureList)

    ourStructureList, counts1 = np.unique(ourStructureList, return_counts=True)
    ourStructureList = ourStructureList[np.argsort(counts1)[-1::-1]]
    ourStructureList = ourStructureList[:20]
    
    
    structureBest = ''
    lossBest = -5000
    lossList = []
    structureList = []

    for structureOurs in ourStructureList:

        structureOurs_string = structureOurs

        structureOurs = list(structureOurs)

        #print (' '.join(structureOurs))

        #print (structureOurs)

        structureMatrix_our = structureToMatrix(structureOurs)


        filledTable_new, sequenceTable_new = generateNeural(sequence, Nstate, structureMatrix_our, model, restriction=(edgeStandard, edgeCombine))
        fullEnd = filledTable_new[0, -1]
        seqEnd = sequenceTable_new[0, -1]
        loss = torch.logsumexp(seqEnd, dim=0) - torch.logsumexp(fullEnd, dim=0)
        #print (loss)

        loss = loss.data.numpy()

        lossList.append(loss)
        structureList.append(structureOurs_string)

        if loss > lossBest:
            lossBest = loss
            structureBest = structureOurs
                


        
        structureBest = list(structureBest)
    
    print (''.join(structureBest))





import sys

if sys.argv[1] == 'pred':

    list1 = sys.argv

    if sys.argv[2] == '-m':
        modelname = sys.argv[3]
        sequence = sys.argv[4]
    else:
        modelname = './14.pt'
        sequence = sys.argv[2]
        

    
    givePrediction(sequence, modelname)

if sys.argv[1] == 'train':

    modelname = sys.argv[2]
    realTrainNeural(modelname)


#python3 RNApred.py pred GGGGCCUUAGCUCAGCUGGGAGAGCGCCUGCUUUGCACGCAGGAGGUCAGCGGUUCGAUCCCGCUAGGCUCCA
#python3 RNApred.py train ./15.pt