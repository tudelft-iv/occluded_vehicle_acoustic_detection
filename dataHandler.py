import pickle
import os
import sys
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from sklearn import preprocessing, metrics

import utilities
import videoPlayer as vp

class DataHandler:

    def __init__(self, basePath=None, csvPath=None, classifierPath=None, showViz=False, store=False, axisLabels=False, baseOutputPath=None):

        if store and baseOutputPath is None:
            raise ValueError("Cannot store files without output Path!")

        if classifierPath is not None:
            self.classifier = pickle.load(open(classifierPath, 'rb'))
        else:
            self.classifier = None

        self.micArray = utilities.loadMicarray()

        self.store = store
        self.showViz = showViz
        self.axisLabels = axisLabels
        self.basePath = basePath
        self.baseOutputPath = baseOutputPath
        if baseOutputPath is not None:
            utilities.makeDirectory(baseOutputPath)

        if basePath is not None:
            self.logTable = pd.read_csv(os.path.join(basePath, 'DataLog.csv'))
            if not all(elem in self.logTable.columns.to_list() for elem in ['ID', 'Environment', 'Class', 'T0']):
                raise ValueError('Provided DataLog.csv doesnt contain all required columns')

            if csvPath is not None:
                filterCsv = pd.read_csv(csvPath)
                if not 'ID' in filterCsv.columns.to_list():
                    raise ValueError('Provided filtering csv doesnt contain the required column ID')
                self.logTable = self.logTable[self.logTable['ID'].isin(filterCsv['ID'])]
                self.logTable = self.logTable.reset_index(drop=True)
        
        if store:
            self.videoPath = os.path.join(baseOutputPath, 'VideoOverlays')
            self.plotPath = os.path.join(baseOutputPath, 'Plots') 
            utilities.makeDirectory(self.videoPath)
            utilities.makeDirectory(self.plotPath)
        else:
            self.videoPath = None
            self.plotPath = None

    def loadResultTable(self, path):
        self.logTable = pickle.load(open(path, 'rb'))

    def setLogTable(self, newTable):
        # join results table with the data table
        self.logTable = self.logTable.join(newTable)
        self.logTable['T0'] -= 10

    def padData(self):

        Tall = self.logTable['T0'].max()
        Lall = self.logTable['VideoLength'].max()
        Dall = Lall - Tall

        # iterate over all rows, we want to pad first three elements, features confidences and the classifcation results
        for index, rows in self.logTable.iterrows():
            Tcurr = rows['T0']
            Lcurr = rows['VideoLength']
            Dcurr = Lcurr - Tcurr

            padBefore = Tall - Tcurr
            padAfter = Dall - Dcurr

            # check if padding was already carried out and then pad the data
            if rows['ClassificationResults'].shape[0] == Lcurr:
                self.logTable.at[index, 'ClassificationResults'] = np.pad(rows['ClassificationResults'], ((padBefore, padAfter), (0,0)))
                self.logTable.at[index, 'ConfidenceResults'] = np.pad(rows['ConfidenceResults'], ((padBefore, padAfter), (0,0)))
                self.logTable.at[index, 'Features'] = np.pad(rows['Features'], ((padBefore, padAfter), (0,0)))
                self.logTable.at[index, 'VisualDetections'] = np.pad(rows['VisualDetections'], ((padBefore, padAfter)))

    def postProcessing(self):
        self.padData()
        self.plotConfidences()
        self.plotAbsolutes()
        self.plotVisualBaseline()
        self.plotMeanFeatures()
        plt.show()
        

    def loopOverRecordings(self):
        
        if self.classifier is not None and self.basePath is not None and self.baseOutputPath is not None:
            newColumns = ['Features', 'ConfidenceResults', 'ClassificationResults', 'VisualDetections', 'InferenceTime', 'VideoLength']
            newTable = pd.DataFrame(columns=newColumns)

            tic = time.perf_counter()

            for index, rows in self.logTable.iterrows():
                identifier = str(rows['ID'])
                location = rows['Environment']
                label = rows['Class']
                tZero = rows['T0']

                print('Starting Session {} with class {} at location {} and T0 of {}, number {} out of {}'.format(identifier, label, location, tZero, index+1, self.logTable.shape[0]))
                samplePath = os.path.join(self.basePath, location, label, identifier)

                session = vp.VideoPlayer(samplePath, self.classifier, self.micArray, self.showViz, self.store, self.videoPath)
                featData, confidenceResults, classificationResults, visualDetections, inferenceTime, videoLength = session.loopOverFrames()

                print('Inference ran for {:.2f} seconds for the {:.1f} second long Clip\n\n'.format(inferenceTime, videoLength/10))

                newTable = newTable.append({'Features': featData, 'ConfidenceResults': confidenceResults, 'ClassificationResults': classificationResults, 'VisualDetections': visualDetections, 'InferenceTime': inferenceTime, 'VideoLength': videoLength}, ignore_index=True)

            toc = time.perf_counter()
            print('Inference ran successfully for all recordings in {} [HH:MM:SS]'.format(time.strftime("%H:%M:%S", time.gmtime(int(round(toc-tic))))))

            # append the results of inference to the log table, do post processing and then dump the data
            self.setLogTable(newTable)
            pickle.dump((self.logTable), open(os.path.join(self.baseOutputPath, 'ResultTable.obj'), 'wb'))
            self.postProcessing()
            pickle.dump((self.logTable), open(os.path.join(self.baseOutputPath, 'ResultTable.obj'), 'wb'))

        else:
            print('Classifier, Datapath or output path is missing, cannot loop over Recordings')

    def plotConfidences(self):

        print('Plotting the mean and std of the confidences of each time step')

        # define iterables and time scale
        unSamples = self.logTable.loc[:,'Class'].unique()
        unEnvironment = self.logTable.loc[:,'Environment'].unique()
        unClasses = ['front', 'left', 'none', 'right']
        c = ['blue', 'green', 'black', 'red']

        Tall = self.logTable['T0'].max()
        Lall = self.logTable['VideoLength'].max()
        time = np.linspace(-Tall  / 10, (Lall - Tall) / 10 - 0.1, Lall)

        # prepare the data
        dataList = []
        for i in range(len(unSamples)):
            locationList = []
            for j in range(len(unEnvironment)):
                tableSubset = self.logTable[self.logTable.Environment == unEnvironment[j]]
                stackedDataList = []
                for k in range(len(unClasses)):

                    # stack the data over all samples of one environment
                    stackedData = []
                    for index, rows in tableSubset[tableSubset.Class == unSamples[i]].iterrows():
                        stackedData.append(rows['ConfidenceResults'])
                    stackedData = np.stack(stackedData, axis=2)

                    # get the mean over all present samples, but do not count values in that were empty (0)
                    nonZeroOverClasses = np.count_nonzero(np.sum(stackedData, axis=1), axis=1)
                    trueMean = np.divide(np.sum(stackedData[:,k], axis=1), nonZeroOverClasses, out=np.zeros_like(np.sum(stackedData[:,k], axis=1)), where=nonZeroOverClasses!=0)
                    stackedDataList.append([trueMean, np.std(stackedData[:,k], axis=1)])
                locationList.append(stackedDataList)

            # repeat for all environment types
            stackedDataList = []
            for k in range(len(unClasses)):
                stackedData = []
                for index, rows in self.logTable[self.logTable.Class == unSamples[i]].iterrows():
                    stackedData.append(rows['ConfidenceResults'])
                stackedData = np.stack(stackedData, axis=2)

                nonZeroOverClasses = np.count_nonzero(np.sum(stackedData, axis=1), axis=1)
                trueMean=np.divide(np.sum(stackedData[:,k], axis=1), nonZeroOverClasses, out=np.zeros_like(np.sum(stackedData[:,k], axis=1)), where=nonZeroOverClasses!=0)
                stackedDataList.append([trueMean, np.std(stackedData[:,k], axis=1)])
            locationList.append(stackedDataList)
            dataList.append(locationList)
        unEnvironment = np.append(unEnvironment, 'overall')

        # store plots separately if true
        if self.store:
            confidencePath = os.path.join(self.plotPath, 'ConfidencePlots')
            utilities.makeDirectory(confidencePath)
            for i in range(len(unSamples)):
                for j in range(len(unEnvironment)):
                    fig, axes = plt.subplots()
                    for k in range(len(unClasses)):
                        axes.plot(time, dataList[i][j][k][0], color=c[k], label=unClasses[k])
                        axes.fill_between(time, dataList[i][j][k][0] - dataList[i][j][k][1], dataList[i][j][k][0] + dataList[i][j][k][1], color=c[k], alpha=0.15)
                        axes.grid('on')
                        axes.set_xlim([np.min(time), np.max(time)])
                        axes.set_ylim([0, 1])
                        if self.axisLabels:
                            axes.set_ylabel('Confidence in [0, 1]')
                            axes.set_xlabel('time t in [s]')
                            axes.set_title(unSamples[i] + ' at ' + unEnvironment[j])
                        if j == len(unEnvironment)-1 and unSamples[i] == 'none':
                            axes.legend(loc='upper center', ncol=4, prop={'size': 8.5})
                    fig.set_size_inches((4.449, 2.40157))
                        
                    plt.savefig(os.path.join(confidencePath, 'Confidence_{}_{}.pdf'.format(unEnvironment[j], unSamples[i])), dpi=300, bbox_inches='tight')
                    plt.close(fig)

        # plot on screen if true
        if self.showViz:
            fig, axes = plt.subplots(len(unSamples), len(unEnvironment))
            for i in range(len(unSamples)):
                for j in range(len(unEnvironment)):
                    for k in range(len(unClasses)):
                        axes[i][j].plot(time, dataList[i][j][k][0], color=c[k], label=unClasses[k])
                        axes[i][j].fill_between(time, dataList[i][j][k][0] - dataList[i][j][k][1], dataList[i][j][k][0] + dataList[i][j][k][1], color=c[k], alpha=0.15)
                        axes[i][j].grid('on')
                        axes[i][j].set_xlim([np.min(time), np.max(time)])
                        axes[i][j].set_ylim([0, 1])
                        if self.axisLabels:
                            axes[i][j].set_ylabel('Confidence in [0, 1]')
                            axes[i][j].set_xlabel('time t in [s]')
                            axes[i][j].set_title(unSamples[i] + ' at ' + unEnvironment[j])
                        if j == len(unEnvironment)-1 and unSamples[i] == 'none':
                            axes[i][j].legend(loc='upper center', ncol=2, prop={'size': 8.5})
            plt.tight_layout()
            

    def plotAbsolutes(self):

        print('Plotting the normalized absolute classification results of each time step')

        # define iterables and time scale
        unSamples = self.logTable.loc[:,'Class'].unique()
        unEnvironment = self.logTable.loc[:,'Environment'].unique()
        unClasses = ['front', 'left', 'none', 'right']
        c = ['blue', 'green', 'black', 'red']

        Tall = self.logTable['T0'].max()
        Lall = self.logTable['VideoLength'].max()
        time = np.linspace(-Tall  / 10, (Lall - Tall) / 10 - 0.1, Lall)

        # prepare the data
        dataList = []
        for i in range(len(unSamples)):
            locationList = []
            for j in range(len(unEnvironment)):
                tableSubset = self.logTable[self.logTable.Environment == unEnvironment[j]]
                summedDataList = []
                for k in range(len(unClasses)):

                    # stack the data over all samples of one environment
                    summedData = np.zeros((len(time), len(unClasses)))
                    for index, rows in tableSubset[tableSubset.Class == unSamples[i]].iterrows():
                        summedData += rows['ClassificationResults']

                    # get the mean over all present samples, but do not count values in that were empty (0)
                    trueMean = np.divide(summedData[:,k], np.sum(summedData, axis=1), out=np.zeros_like(summedData[:,k]), where=np.sum(summedData, axis=1)!=0)
                    summedDataList.append(trueMean)
                locationList.append(summedDataList)

            # repeat for all environment types
            summedDataList = []
            for k in range(len(unClasses)):
                summedData = np.zeros((len(time), len(unClasses)))
                for index, rows in self.logTable[self.logTable.Class == unSamples[i]].iterrows():
                    summedData += rows['ClassificationResults']

                trueMean=np.divide(summedData[:,k], np.sum(summedData, axis=1), out=np.zeros_like(summedData[:,k]), where=np.sum(summedData, axis=1)!=0)
                summedDataList.append(trueMean)
            locationList.append(summedDataList)
            dataList.append(locationList)
        unEnvironment = np.append(unEnvironment, 'overall')

        # store plots separately if true
        if self.store:
            confidencePath = os.path.join(self.plotPath, 'AbsolutePlots')
            utilities.makeDirectory(confidencePath)
            for i in range(len(unSamples)):
                for j in range(len(unEnvironment)):
                    fig, axes = plt.subplots()
                    for k in range(len(unClasses)):
                        axes.plot(time, dataList[i][j][k], color=c[k], label=unClasses[k])
                        axes.grid('on')
                        axes.set_xlim([np.min(time), np.max(time)])
                        axes.set_ylim([0, 1])
                        if self.axisLabels:
                            axes.set_ylabel('Normalized classifications in [0, 1]')
                            axes.set_xlabel('time t in [s]')
                            axes.set_title(unSamples[i] + ' at ' + unEnvironment[j])
                        if j == len(unEnvironment)-1 and unSamples[i] == 'negative':
                            axes.legend(loc='upper center', ncol=4, prop={'size': 8.5})
                        
                    plt.savefig(os.path.join(confidencePath, 'Absolute_{}_{}.pdf'.format(unEnvironment[j], unSamples[i])), dpi=300, bbox_inches='tight')
                    plt.close(fig)

        # plot on screen if true
        if self.showViz:
            fig, axes = plt.subplots(len(unSamples), len(unEnvironment))
            for i in range(len(unSamples)):
                for j in range(len(unEnvironment)):
                    for k in range(len(unClasses)):
                        axes[i][j].plot(time, dataList[i][j][k], color=c[k], label=unClasses[k])
                        axes[i][j].grid('on')
                        axes[i][j].set_xlim([np.min(time), np.max(time)])
                        axes[i][j].set_ylim([0, 1])
                        if self.axisLabels:
                            axes[i][j].set_ylabel('Normalized classifications in [0, 1]')
                            axes[i][j].set_xlabel('time t in [s]')
                            axes[i][j].set_title(unSamples[i] + ' at ' + unEnvironment[j])
                        if j == len(unEnvironment)-1 and unSamples[i] == 'negative':
                            axes[i][j].legend(loc='upper center', ncol=2, prop={'size': 8.5})
            plt.tight_layout()
           


    def plotVisualBaseline(self):

        print('Plotting the auditory classification results against the visual baseline')

        le = preprocessing.LabelBinarizer()
        le.fit([0,1,2,3])

        # prepare gt labels
        unClasses = ['front', 'left', 'negative', 'right']

        Tall = self.logTable['T0'].max()
        Lall = self.logTable['VideoLength'].max()
        time = np.linspace(-Tall  / 10, (Lall - Tall) / 10 - 0.1, Lall)
        offset = 15

        gtLeft = np.zeros((Lall, len(unClasses)), dtype=int)
        gtLeft[:Tall+offset,1] = 1
        gtLeft[Tall+1:,0] = 1

        gtRight = np.zeros((Lall, len(unClasses)), dtype=int)
        gtRight[:Tall+offset,3] = 1
        gtRight[Tall+1:,0] = 1

        gtNone = np.zeros((Lall, len(unClasses)), dtype=int)
        gtNone[:,2] = 1

        gtVisualNeg = np.zeros(Lall, dtype=int)
        gtVisualPos = np.ones(Lall, dtype=int)

        # stack labels and results
        stackedLabels = []
        stackedResults = []
        stackedVisualLabel = []
        stackedVisualResult = []

        for index, rows in self.logTable.iterrows():
            if rows['Class'] == 'left':
                stackedLabels.append(gtLeft)
                stackedVisualLabel.append(gtVisualPos)
            elif rows['Class'] == 'right':
                stackedLabels.append(gtRight)
                stackedVisualLabel.append(gtVisualPos)
            else:
                stackedLabels.append(gtNone)
                stackedVisualLabel.append(gtVisualNeg)

            stackedResults.append(rows['ClassificationResults'])
            stackedVisualResult.append(rows['VisualDetections'])
            

        stackedLabels = np.stack(stackedLabels)
        stackedResults = np.stack(stackedResults)
        stackedVisualLabel = np.stack(stackedVisualLabel)
        stackedVisualResult = np.stack(stackedVisualResult)

        # print(stackedVisualResult)
        # 
        visualAccuracy = []
        auditoryAccuracy = []

        for i in range(stackedLabels.shape[1]):
            # Make sure only elements are considered that contributed and filter those, i.e. results do not show zero just from padding
            idx = np.where(np.sum(stackedResults[:,i,:], axis=1) != 0)[0]
            stackedLabelsStep = stackedLabels[idx,i,:]
            stackedResultsStep = stackedResults[idx,i,:]
            stackedVisualLabelStep = stackedVisualLabel[idx,i]
            stackedVisualResultStep = stackedVisualResult[idx,i]

            # get indexes from label matrix to extract from Result matrix
            idxLeft  = np.where(stackedLabelsStep[:,1] == 1)[0]
            idxRight = np.where(stackedLabelsStep[:,3] == 1)[0]
            idxNone  = np.where(stackedLabelsStep[:,2] == 1)[0]  
            idxFront = np.where(stackedLabelsStep[:,0] == 1)[0]

            # first deal with visual detections
            tn, _, _, tp = metrics.confusion_matrix(stackedVisualLabelStep, stackedVisualResultStep, labels=[0,1]).ravel()

            visualAccuracy.append((tn + tp) / len(stackedVisualLabelStep))

            # in offset area manually calculate TPs
            if i > Tall and i <= Tall+offset:
                totalSamples = len(idxLeft) + len(idxRight) + len(idxNone)

                # define tp as correct whenever front or left/right is correct
                tpa = np.array([np.sum(stackedResultsStep[idxLeft,0]) + np.sum(stackedResultsStep[idxLeft,1]), 
                        np.sum(stackedResultsStep[idxRight,0]) + np.sum(stackedResultsStep[idxRight,3]),
                        np.sum(stackedResultsStep[idxNone, 2])])

                
            else:
                totalSamples = len(idxLeft) + len(idxRight) + len(idxNone) + len(idxFront)
                mcm = metrics.multilabel_confusion_matrix(stackedLabelsStep, stackedResultsStep, labels=[0,1,2,3])
                tpa = mcm[:, 1, 1]
                idx_two = np.where(np.sum(stackedLabelsStep, axis=1) == 2)
            
            auditoryAccuracy.append(np.sum(tpa) / totalSamples)

        visualAccuracy = np.array(visualAccuracy)
        auditoryAccuracy = np.array(auditoryAccuracy)


        # now plot
        fig, axes = plt.subplots()
        transitionArea = np.zeros((Lall))
        transitionArea[Tall+1:Tall+offset] = 1

        axes.fill_between(time, transitionArea, 0, color='k', step='mid', alpha=0.2)

        axes.plot(time, visualAccuracy, label='Faster R-CNN (visual)', linewidth=2.5)
        axes.plot(time, auditoryAccuracy, 'peru', label='ours (acoustic)', linewidth=2.5)
        axes.grid('on')
        axes.set_xlim([np.min(time), np.max(time)])
        axes.set_ylim([0.4, 1])
        if self.axisLabels:
            axes.set_xlabel('time t in [s]')
            axes.set_ylabel('Accuracy in [0, 1]')
            axes.set_title('Visual Baseline Comparison')
        axes.legend(loc='upper left', ncol = 1, prop={'size': 10}) 
        if self.store:
            baselinePath = os.path.join(self.plotPath, 'VisualBaselineComparison')
            utilities.makeDirectory(baselinePath)
            fig.set_size_inches((1.8*4.449, 1.8*1.40157))
            plt.savefig(os.path.join(baselinePath, 'VisualBaselineComparison.pdf'), dpi=300, bbox_inches='tight')
        if self.showViz:
            plt.tight_layout()
        else:
            plt.close(fig)


    def plotMeanFeatures(self):

        print('Plotting the mean feature vector of each time step')

        # define iterables and time scale
        unSamples = self.logTable.loc[:,'Class'].unique()
        unEnvironment = self.logTable.loc[:,'Environment'].unique()

        Tall = self.logTable['T0'].max()
        Lall = self.logTable['VideoLength'].max()
        time = np.linspace(-Tall  / 10, (Lall - Tall) / 10 - 0.1, Lall)
        offset = 15

        maxValue = 0
        minValue = 100

        # prepare the data
        dataList = []
        for i in range(len(unSamples)):
            locationList = []
            for j in range(len(unEnvironment)):
                tableSubset = self.logTable[self.logTable.Environment == unEnvironment[j]]

                # stack the data over all samples of one environment
                stackedData = []
                for index, rows in tableSubset[tableSubset.Class == unSamples[i]].iterrows():
                    stackedData.append(rows['Features'])
                stackedData = np.stack(stackedData, axis=2)

                # get the mean over all present samples, but do not count values in that were empty (0)
                nonZeroOverClasses = np.count_nonzero(stackedData, axis=2)
                trueMean = np.divide(np.sum(stackedData, axis=2), nonZeroOverClasses, out=np.zeros_like(np.sum(stackedData, axis=2)), where=nonZeroOverClasses!=0)
                if trueMean.max() > maxValue:
                    maxValue = trueMean.max()
                if trueMean.min() < minValue:
                    minValue = trueMean.min()
                locationList.append([trueMean, np.std(stackedData)])

            # repeat for all environment types
            stackedData = []
            for index, rows in self.logTable[self.logTable.Class == unSamples[i]].iterrows():
                stackedData.append(rows['Features'])
            stackedData = np.stack(stackedData, axis=2)

            nonZeroOverClasses = np.count_nonzero(stackedData, axis=2)
            trueMean=np.divide(np.sum(stackedData, axis=2), nonZeroOverClasses, out=np.zeros_like(np.sum(stackedData, axis=2)), where=nonZeroOverClasses!=0)
            if trueMean.max() > maxValue:
                maxValue = trueMean.max()
            if trueMean.min() < minValue:
                minValue = trueMean.min()

            locationList.append([trueMean, np.std(stackedData)])
            dataList.append(locationList)
        unEnvironment = np.append(unEnvironment, 'overall')

        # store plots separately if true
        if self.store:
            confidencePath = os.path.join(self.plotPath, 'FeaturePlots')
            utilities.makeDirectory(confidencePath)
            for i in range(len(unSamples)):
                for j in range(len(unEnvironment)):
                    fig, axes = plt.subplots()
                    axes.imshow(np.transpose(dataList[i][j][0])[0:29,:], cmap='gray', vmin=minValue, vmax=maxValue)
                    axes.vlines([Tall, Tall+offset], 0, 29, colors=['blue', 'red'], linestyles='dashed')
                    
                    axes.set_xticks(np.arange(len(time)))
                    axes.set_xticklabels(time)
                    axes.set_yticks(np.arange(30))
                    axes.set_yticklabels(np.linspace(-90, 90, 30).astype(int))

                    everyNth = 10
                    for n, label in enumerate(axes.xaxis.get_ticklabels()):
                        if n % everyNth != 0:
                            label.set_visible(False)

                    everyNth = 3
                    for n, label in enumerate(axes.yaxis.get_ticklabels()):
                        if n % everyNth != 0:
                            label.set_visible(False)

                    axes.set_ylim([0, 29])
                    
                    if self.axisLabels:
                        axes.set_title(unSamples[i] + ' at ' + unEnvironment[j])
                        axes.set_xlabel('Angle in [deg]')
                        axes.set_ylabel('time t in [s]')
                        
                    fig.set_size_inches((4.449, 2.40157))

                    plt.savefig(os.path.join(confidencePath, 'Feature_{}_{}.pdf'.format(unEnvironment[j], unSamples[i])), dpi=300, bbox_inches='tight')
                    plt.close(fig)

        # plot on screen if true
        if self.showViz:
            fig, axes = plt.subplots(len(unSamples), len(unEnvironment))
            for i in range(len(unSamples)):
                for j in range(len(unEnvironment)):

                    axes[i][j].imshow(np.transpose(dataList[i][j][0])[0:29,:], cmap='gray', vmin=minValue, vmax=maxValue)
                    axes[i][j].vlines([Tall, Tall+offset], 0, 29, colors=['blue', 'red'], linestyles='dashed')

                    axes[i][j].set_xticks(np.arange(len(time)))
                    axes[i][j].set_xticklabels(time)
                    axes[i][j].set_yticks(np.arange(30))
                    axes[i][j].set_yticklabels(np.linspace(-90, 90, 30).astype(int))

                    everyNth = 10
                    for n, label in enumerate(axes[i][j].xaxis.get_ticklabels()):
                        if n % everyNth != 0:
                            label.set_visible(False)

                    everyNth = 3
                    for n, label in enumerate(axes[i][j].yaxis.get_ticklabels()):
                        if n % everyNth != 0:
                            label.set_visible(False)

                    axes[i][j].set_ylim([0, 29])
                    
                    if self.axisLabels:
                        axes[i][j].set_title(unSamples[i] + ' at ' + unEnvironment[j])
                        axes[i][j].set_xlabel('Angle in [deg]')
                        axes[i][j].set_ylabel('time t in [s]')

            plt.tight_layout()