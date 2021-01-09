import sys
import argparse

import dataHandler as dh

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
        This script tests the classifier result on extracted data in the common format and stores the results in a table.
        Plots can be generated and stored. A visualization of the inference with an overlay can be shown and stored as a Video.

        Examples:
        minimum:
        python3 timeHorizonInference.py --input [INPUT] --class [CLASSIFIER] --output [OUTPUT] 

        full:
        python3 timeHorizonInference.py --input [INPUT] --class [CLASSIFIER] --output [OUTPUT] --csv [CSV] --vis --store --axis-labels
    """
    parser = ExtractorArgsParser(description='Tests the inference of a trained classifier on the full lengths clips and audio', usage=usage)

    groupRequired = parser.add_argument_group('Required Arguments')
    groupRequired.add_argument('--input', dest='input', help='The input path of the extraction folder of locations, make sure DataLog.csv exists', required=True, default=None)
    groupRequired.add_argument('--class', dest='classifier', help='The path to the classifier file to use. As pickle .obj', required=True, default=None)
    groupRequired.add_argument('--output', dest='output', help='The output directory where the results shall be stored', required=True, default=None)

    groupOptional = parser.add_argument_group('Filter, visualisation and storing')
    groupOptional.add_argument('--csv', dest='csv', help='The path to a csv definining the recording IDs that should be used', required=False, default=None)
    groupOptional.add_argument('--vis', action='store_true', dest='vis', help='If passed there will be a live visualization of the classification and consecutive plotting of the figures', required=False)
    groupOptional.add_argument('--store', action='store_true', dest='store', help='If passed, the overlays and plots will be stored', required=False)
    groupOptional.add_argument('--axis-labels', action='store_true', dest='axisLabels', help='If passed, axis labels will generated on the plots', required=False)


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

    handler = dh.DataHandler(basePath=parsed.input, csvPath=parsed.csv, classifierPath=parsed.classifier, showViz=parsed.vis, store=parsed.store, axisLabels=parsed.axisLabels, baseOutputPath=parsed.output)

    handler.loopOverRecordings()

if __name__ == "__main__":
    main()
