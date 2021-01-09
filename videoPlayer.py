import cv2
import numpy as np
import os
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import signal
import pyroomacoustics as pra

import time
from termcolor import colored

import utilities

class VideoPlayer:

    def __init__(self, inputPath, classifier, micArray, showViz=False, store=False, outputPath=None):
        
        self.inputPath = inputPath
        self.outputOverlayPath = outputPath
        self.labelDict = ['front', 'left', 'none', 'right']


        self.showViz = showViz
        self.store = store

        # load the audio data, classifier and define audio feature parameters
        self.sampleRate, self.audioData = wavf.read(os.path.join(self.inputPath, 'out_multi.wav'))
        self.classifier = classifier

        self.micArray = micArray
        self.nfft = 2*256
        self.resolution = 30
        self.freqRange = [50,1500]
        
        # video parameters serve as iterator
        self.fps = 10
        self.nFrames = round(self.audioData.shape[0] / self.sampleRate * self.fps ) - self.fps + 1

        # load the visual detections in
        if os.path.isfile(os.path.join(inputPath, 'camera_baseline_detections.json')):
            self.visualDetectionsBBox = utilities.detectionFolder(self.inputPath, score_threshold=0.75, height_threshold=100)
            self.visualDetectionsBBox = [element or 0 for element in self.visualDetectionsBBox]
            self.visualDetectionsBBox = self.visualDetectionsBBox[self.fps:]
        else:
            self.visualDetectionsBBox = np.zeros(self.nFrames, dtype=int)
        
        self.visualDetectionsBinary = np.array(self.visualDetectionsBBox, dtype=bool)

        # Prepare visualization and output files
        if self.showViz or self.store:

            # Set scale for the video
            # TODO: Only change here
            self.scale = 0.6

            # load the video and make some checks
            if not os.path.isfile(os.path.join(inputPath, 'ueye_stereo_vid.mp4')):
                raise ValueError("Cannot find Video, make sure the Video exists")

            self.cap = cv2.VideoCapture(os.path.join(inputPath, 'ueye_stereo_vid.mp4'))
            fpsVideo = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fps)
            nFramesVideo = int(round(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - self.fps))

            if fpsVideo != self.fps or nFramesVideo != self.nFrames:
                raise ValueError("The loaded video parameters (fps or length) are not compatible with the audio data!")
            
            self.width = int(self.cap.get(3))
            self.height = int(self.cap.get(4))

            self.newWidth = int(self.scale * self.width)
            self.newHeight = int(self.scale * self.height)

            # Prepare the figure for redraw
            self.fig, ax= plt.subplots(figsize=(self.scale * 10, self.scale * 4.58)) 
            
            x1 = np.linspace(1.,60., 60)
            ang = np.concatenate((np.linspace(-90,90,30, dtype=np.int), np.linspace(-90,90,30, dtype=np.int)))
            ax.set_ylim([0.05, 0.4])
            ax.grid('on')
            ax.set_xlabel('angular bins of L=2 segments [deg]', fontsize=self.scale * 16.6, fontfamily='serif')
            ax.set_ylabel('feature intensity', fontsize=self.scale * 16.6, fontfamily='serif')
            ax.tick_params(labelsize=10* self.scale)
            self.fig.tight_layout(pad=0)

            # set appropriate ticks
            ax.set_xticks(np.arange(len(ang)))
            ax.set_xticklabels(ang.astype(int))
            ax.axvline(x=29.5)
            self.line1, = ax.plot(x1, 'ko-')

            # Remove unnecassary ticks for readability
            everyNth = 2
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % everyNth != 0:
                    label.set_visible(False)

            # Image Overlay Parameters
            self.confidenceHeight = round(self.scale * 33)
            self.confidenceWidth = round(self.scale * 167)
            self.confidenceOffset = round(self.scale * 50)
            self.confidenceTextOffset = round(self.scale * 175)

            self.resultArrowY = self.newHeight // 2 + round(self.scale * 250)
            self.resultArrowLeft = self.newWidth // 2 - round(self.scale * 167)
            self.resultArrowRight = self.newWidth // 2 + round(self.scale * 167)

            self.resultTextY = self.newHeight // 2 + round(self.scale * 417)
            self.resultTextX = self.newWidth // 2 - round(self.scale * 83)

            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.fontScale = 1.67 * self.scale
            self.lineType = round(4 * self.scale)

            self.colorFront = (255,0,0)
            self.colorLeft = (0,255,0)
            self.colorRight = (0,0,255)
            self.colorNone = (0,0,0)

        if self.store:
            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file. cv2.VideoWriter_fourcc('M','J','P','G'), cv2.VideoWriter_fourcc(*'XVID')
            self.outputVideoPath = os.path.join(outputPath, 'out_video.avi')
            self.outputStereoPath = os.path.join(outputPath, 'out_stereo.wav')
            self.outputOverlayPath = os.path.join(outputPath, os.path.basename(os.path.normpath(inputPath)) + '.mp4')

            wavf.write(self.outputStereoPath, self.sampleRate, self.audioData[self.sampleRate:,[5,25]])
            self.out = cv2.VideoWriter(self.outputVideoPath, cv2.VideoWriter_fourcc('M','J','P','G'), fps=self.fps, frameSize=(self.newWidth, self.newHeight))

    def loopOverFrames(self):
        printColor = ['blue', 'green', 'white', 'red']

        # initialize pass through lists
        classificationResults = []
        confidenceResults = []
        featData = []

        # get timer ready
        tic = time.perf_counter()

        # extract the stft from parameters
        stftContainer = []
        for i in range(self.audioData.shape[1]):
            _, _, stft = signal.stft(self.audioData[:,i], self.sampleRate, nperseg=self.nfft)
            stftContainer.append(stft)
        stftContainer = np.stack(stftContainer, axis=0)

        # help out not fitting window sizes in the STFT by a more distributed indexing
        nHops = self.nFrames + self.fps - 1
        stepSize, nOdd = np.divmod(stftContainer.shape[2], nHops)
        tempArray = stepSize+1 * np.ones(nOdd, dtype=int)
        tempArray = np.array_split(tempArray, nHops - nOdd + 1)

        idxEqualizer = np.concatenate((np.array([0]), tempArray[0]))
        for i in range(nHops - nOdd):
            idxEqualizer = np.concatenate((idxEqualizer, np.array([stepSize]), tempArray[i+1]))
        
        idxEqualizer = np.cumsum(idxEqualizer)

        windowSize = int(round(stftContainer.shape[2] * self.fps / (self.nFrames + self.fps)))

        # prepare DataProcessor
        doaProcessor = pra.doa.algorithms['SRP'](self.micArray.transpose(), self.sampleRate, self.nfft, azimuth=np.linspace(-90.,90., self.resolution)*np.pi/180)

        # loop over all frames and capture frame by frame
        for frameCurr in range(self.nFrames):

            # extract current moment and run feature extraction
            container = np.array_split(stftContainer[:,:, idxEqualizer[frameCurr] : idxEqualizer[frameCurr] + windowSize], 2, axis=2)

            featSingle = []
            for i in range(2):
                doaProcessor.locate_sources(container[i], freq_range=self.freqRange)
                featSingle.append(doaProcessor.grid.values)

            featSingle = np.concatenate(featSingle)
            featData.append(featSingle)

            confEstimate = self.classifier.predict_proba(featSingle.reshape(1,-1))[0]
            resultEstimate = np.argmax(confEstimate)

            # Print results
            if frameCurr > 0:
                print ("\033[A                                                                                                       \033[A")
            print('Estimated class', colored('{}'.format(self.labelDict[resultEstimate]), printColor[resultEstimate]), 'at Frame {} with confidence {:.2f}'.format(frameCurr + self.fps, confEstimate[resultEstimate]), end='')
            if self.visualDetectionsBinary[frameCurr]:
                print(' and found visual detection with confidence {:.2f}'.format(self.visualDetectionsBBox[frameCurr][0][1]))
            else:
                print('')

            confidenceResults.append(confEstimate)
            classificationResults.append(resultEstimate)

            #rewrite and add the new data to the plots
            if self.showViz or self.store:
                # load frame
                _, frame = self.cap.read()

                # redraw canvas
                self.line1.set_ydata(featSingle)
                self.fig.canvas.draw()

                # convert canvas to image
                img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # resize image
                frame = cv2.resize(frame, (self.newWidth, self.newHeight), interpolation = cv2.INTER_AREA) 

                # put plot on the image and draw the classification result on it
                frame[:img.shape[0],-img.shape[1]:,:] = img

                if self.visualDetectionsBinary[frameCurr]:
                    frame = self.drawVisualDetection(frame, frameCurr)
                frame = self.drawOnImage(frame, confEstimate, resultEstimate, featSingle)

            # draw and or store the result if true
            if self.showViz:
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)

            if self.store:
                self.out.write(frame)
     
        toc = time.perf_counter()

        # Closes all the frames
        if self.showViz or self.store:
            cv2.destroyAllWindows()
            plt.close('all')

        if self.store:
            self.out.release()
            os.system('ffmpeg -y -loglevel quiet -i {0} -i {1} -c:v copy -c:a aac {2}'.format(self.outputVideoPath, self.outputStereoPath, self.outputOverlayPath))
            os.system('rm {0} & rm {1}'.format(self.outputVideoPath, self.outputStereoPath))

        le = preprocessing.LabelBinarizer()
        le.fit([0,1,2,3])
        return np.stack(featData), np.stack(confidenceResults), le.transform(np.stack(classificationResults)), self.visualDetectionsBinary.astype(int), toc-tic, self.nFrames


    def drawVisualDetection(self, overlay, index):
        # Draw each bounding box on screen
        for element in self.visualDetectionsBBox[index]:
            if element != 0:
                bbox = np.array(element[0])*self.scale
                bbox = bbox.astype(int)
                cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colorFront, 2*round(self.fontScale))
                cv2.putText(overlay,'camera baseline', (bbox[0], bbox[1]-2*self.lineType), self.font, 0.5 * self.fontScale, self.colorFront,  self.lineType)

        return overlay


    def drawOnImage(self, overlay, confEstimate, resultEstimate, feature):

        # First draw confidences in boxes in top left corner
        pos_d_x = self.confidenceHeight
        pos_d_y = self.confidenceHeight

        cv2.rectangle(overlay, (pos_d_x, pos_d_y), (pos_d_x + int(self.confidenceWidth*confEstimate[1]), pos_d_y + self.confidenceHeight), self.colorLeft, -1)
        cv2.rectangle(overlay, (pos_d_x, pos_d_y), (pos_d_x + self.confidenceWidth, pos_d_y + self.confidenceHeight), (0,0,0), round(self.fontScale))
        cv2.putText(overlay,'left', (pos_d_x + self.confidenceTextOffset, pos_d_y + self.confidenceHeight), self.font, self.fontScale, self.colorLeft, self.lineType)

        pos_d_y = pos_d_y + self.confidenceOffset


        cv2.rectangle(overlay, (pos_d_x, pos_d_y), (pos_d_x + int(self.confidenceWidth*confEstimate[0]), pos_d_y + self.confidenceHeight), self.colorFront, -1)
        cv2.rectangle(overlay, (pos_d_x, pos_d_y), (pos_d_x + self.confidenceWidth, pos_d_y + self.confidenceHeight), (0,0,0), round(self.fontScale))
        cv2.putText(overlay,'front', (pos_d_x + self.confidenceTextOffset, pos_d_y + self.confidenceHeight), self.font, self.fontScale, self.colorFront, self.lineType)

        pos_d_y = pos_d_y + self.confidenceOffset

        cv2.rectangle(overlay, (pos_d_x, pos_d_y), (pos_d_x + int(self.confidenceWidth*confEstimate[3]), pos_d_y + self.confidenceHeight), self.colorRight, -1)
        cv2.rectangle(overlay, (pos_d_x, pos_d_y), (pos_d_x + self.confidenceWidth, pos_d_y + self.confidenceHeight), (0,0,0), round(self.fontScale))
        cv2.putText(overlay,'right', (pos_d_x + self.confidenceTextOffset, pos_d_y + self.confidenceHeight), self.font, self.fontScale, self.colorRight, self.lineType)

        pos_d_y = pos_d_y + self.confidenceOffset

        cv2.rectangle(overlay, (pos_d_x, pos_d_y), (pos_d_x + int(self.confidenceWidth*confEstimate[2]), pos_d_y + self.confidenceHeight), self.colorNone, -1)
        cv2.rectangle(overlay, (pos_d_x, pos_d_y), (pos_d_x + self.confidenceWidth, pos_d_y + self.confidenceHeight), (0, 0, 0), round(self.fontScale))
        cv2.putText(overlay,'none', (pos_d_x + self.confidenceTextOffset, pos_d_y + self.confidenceHeight), self.font, self.fontScale, self.colorNone, self.lineType)

        # now draw classification result big on screen
        if resultEstimate == 0:
            # convert doa feature to the actual angle of the detection
            # TODO: don't use hardcoded range of the feature
            angles = np.linspace(-np.pi / 2, np.pi / 2, 30)
            cur_angle = angles[np.where(feature[30:] == np.max(feature[30:]))]
            x_off = int(np.sin(cur_angle) * self.confidenceWidth)
            y_off = int(-np.cos(cur_angle) * self.confidenceWidth)

            cv2.arrowedLine(overlay, (self.newWidth // 2, self.resultArrowY), (self.newWidth // 2 + x_off, self.resultArrowY + y_off), self.colorFront, 2 * self.lineType)
            cv2.putText(overlay,'front', (self.resultTextX, self.resultTextY), self.font, 2 * self.fontScale, self.colorFront, self.lineType)
        elif resultEstimate == 1:
            cv2.arrowedLine(overlay, (self.resultArrowRight, self.resultArrowY), (self.resultArrowLeft, self.resultArrowY), self.colorLeft, 2 * self.lineType)
            cv2.putText(overlay,'left', (self.resultTextX, self.resultTextY), self.font, 2 * self.fontScale, self.colorLeft, self.lineType)
        elif resultEstimate == 3:
            cv2.arrowedLine(overlay, (self.resultArrowLeft, self.resultArrowY), (self.resultArrowRight, self.resultArrowY), self.colorRight, 2 * self.lineType)
            cv2.putText(overlay,'right', (self.resultTextX, self.resultTextY), self.font, 2 * self.fontScale, self.colorRight, self.lineType)
        else:
            cv2.putText(overlay,'none', (self.resultTextX, self.resultTextY), self.font, 2 * self.fontScale, self.colorNone, self.lineType)

        return overlay