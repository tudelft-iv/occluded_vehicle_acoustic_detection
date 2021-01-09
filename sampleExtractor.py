import sys
import argparse
import os
import pandas as pd
from scipy.io import wavfile as wavf
import utilities


def parseArgs():
    class ExtractorArgsParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sys.exit(2)
        def format_help(self):
            formatter = self._get_formatter()
            formatter.add_text(self.description)
            formatter.add_usage(self.usage, self._actions,
                                self._mutually_exclusive_groups)
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.add_text(self.epilog)

            return formatter.format_help()
    usage = """
        This script extracts samples from the dataset in the folder structure defined in the readme of this repository.
    """
    parser = ExtractorArgsParser(description='', usage=usage)
    parser.add_argument('--input', dest='input', help='The input path pointing to the top level folder of the dataset', required=True, default=None)
    parser.add_argument('--output', dest='output', help='The output path where the samples shall be stored', required=True, default=None)
    parser.add_argument('--length', dest='length', help='The length of the sample that shall be extracted, (default 1s)', default=1)


    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(2)

    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)

    return parsed


def main():
    parsed = parseArgs()

    # load datalog table and prepare directories 
    logTable = pd.read_csv(os.path.join(parsed.input, 'DataLog.csv'))

    utilities.makeDirectory(os.path.join(parsed.output, 'front'))
    utilities.makeDirectory(os.path.join(parsed.output, 'left'))
    utilities.makeDirectory(os.path.join(parsed.output, 'none'))
    utilities.makeDirectory(os.path.join(parsed.output, 'right'))

    dynamic = ['DA1', 'DA2', 'DB1', 'DB2', 'DB3']
    sampleLength = float(parsed.length)

    frontFrame = []
    recordingID = [] 
    frontCounter = 0

    # loop through table
    for index, row in logTable.iterrows():
        inputPath = os.path.join(parsed.input, row['Environment'], row['Class'], row['ID'])
        sampleRate, audioData = wavf.read(os.path.join(inputPath, 'out_multi.wav'))
        fps = 10

        recordingID.append(index)        
        tSeconds = row['T0'] / fps

        if index > 0:
            print ("\033[A                                                                                \033[A")
        print('Extracting sample ID {} of class {}, {} percent complete'.format(row['ID'], row['Class'], round((index+1) / logTable.shape[0] * 100, 1)))

        # if dynamic sample extract data around tau0, i.e. [tau0-0.5, tau0+0.5], afterwards write to file
        if row['Environment'] in dynamic:
            tSeconds = row['T0'] / fps + 0.5
            
        audioDataSegment = audioData[int((tSeconds - sampleLength)  * sampleRate) : int(tSeconds  * sampleRate), :]
        wavf.write(os.path.join(parsed.output, row['Class'], row['ID'] + '.wav'), sampleRate, audioDataSegment)

        # if recording is left or right, extract a front sample at T0+0.5 to t0+1.5 
        if row['Class'] == 'left' or row['Class'] == 'right':
            audioDataFront = audioData[int((tSeconds + 1.5 - sampleLength)  * sampleRate) : int((tSeconds + 1.5)  * sampleRate), :]

            frontID = '0' + row['ID'][1:5] + str(frontCounter).zfill(4)
            wavf.write(os.path.join(parsed.output, 'front', frontID  + '.wav'), sampleRate, audioDataFront)
            frontFrame.append([frontID, row[1], 'front', row[3], index] )

            frontCounter += 1
    
    # create sample log file with additional unique recording id and front samples 
    logTable['Recording ID'] = recordingID
    logTable = logTable.append(pd.DataFrame(frontFrame, columns=['ID', 'Environment', 'Class', 'T0', 'Recording ID'] ))

    logTable.to_csv(os.path.join(parsed.output, 'SampleLog.csv'), index=False)

if __name__ == "__main__":
    main()