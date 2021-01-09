import sys
import argparse
import utilities
import metrics

import sklearn as skl
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pdb

class Expts:

    def __init__(self, data, output_folder, data_aug=True, classifier=None, random_state=0):

        self.classifier = classifier
        self.data = data
        self.random_state = random_state   #set this to None for randomizing the data shuffle and fold split.
        self.paper_metrics_only = True
        self.output_folder = output_folder
        self.data_aug = data_aug
        
        #feature related parameters
        self.srp_dict = dict(res=data.resolution, nsegs=data.L)

        #for cross validation expt.
        self.folds = 5
        
        if classifier is None:
            self.classifier = SVC(kernel='linear', C=1, random_state=self.random_state)


    def run_train_and_save_classifier(self, save_classifier=True, locs_in=["SAB"], data_split_ratio=0.15):
        
        locations = utilities.get_locations(locs_in)

        print(" \n\n--- Train a classifier for locations: {}\n".format(locations))
        print("Data split into train and test set at ratio: {}\n".format(data_split_ratio))

        data_in = self.data.get_data()
        data_in = data_in[data_in["Environment"].isin(locations)]

        label_encoder, pipeline = utilities.prepare_skl_interface(data_in, self.classifier)

        # stratified split data into train and test - stratification considers location and the class labels

        temp_df = data_in[['Environment', 'Recording ID', 'Class']]
        temp_df = temp_df.drop(temp_df[temp_df['Class'] == 'front'].index)   # just to avoid repeated Recording IDs

        train_bags, test_bags = train_test_split(temp_df, 
                                                test_size=data_split_ratio,
                                                random_state=self.random_state,
                                                stratify=temp_df[['Environment', 'Class']])

        # check if samples from same recordings are present in both train and test
        for bag in list(test_bags['Recording ID']):
            if bag in list(train_bags['Recording ID']):
                print("Error: {}".format(bag))
       
        train_data = data_in[data_in['Recording ID'].isin(train_bags['Recording ID'])]
        test_data = data_in[data_in['Recording ID'].isin(test_bags['Recording ID'])]
        
        accuracy, conf_mat = utilities.train_and_test(train_data, test_data, pipeline, label_encoder, self.srp_dict, 
                                                        save_cls=save_classifier, out_folder=self.output_folder)

        all_metrics = {"overall_accuracy" : (accuracy, 0), 
                       "per_class_accuracy": (metrics.getPCaccuracy(conf_mat), np.zeros(4)), 
                       "per_class_precision": (metrics.getPCPrecision(conf_mat), np.zeros(4)), 
                       "per_class_recall": (metrics.getPCRecall(conf_mat), np.zeros(4)), 
                       "per_class_iou": (metrics.getPCIoU(conf_mat), np.zeros(4))}
        
        metrics.print_metrics(all_metrics, self.paper_metrics_only)


    def run_cross_validation(self, locs_in=["SAB"]):
        
        print(" \n\n--- Cross Validation for locations: {}\n".format(locs_in))

        locations = utilities.get_locations(locs_in)

        data_in = self.data.get_data()
        data_in = data_in[data_in["Environment"].isin(locations)]

        label_encoder, pipeline = utilities.prepare_skl_interface(data_in, self.classifier)

        # shuffle with random seed if specified
        if self.random_state is not None:
            data_in_shuffled = skl.utils.shuffle(data_in, random_state=self.random_state)
        else:
            data_in_shuffled = skl.utils.shuffle(data_in)

        # get metrics
        output_metrics = utilities.cross_validation(pipeline, self.folds, data_in_shuffled, label_encoder, self.srp_dict, data_aug=self.data_aug)

        metrics.print_metrics(output_metrics, self.paper_metrics_only)

    def run_generalisation(self, train_locs=["DA"], test_locs=["DB"], save_classifier=True):

        print("\n\n --- Generalization across locations --- \n")
        print("Locations in train set: {}".format(train_locs))
        print("Locations in test set: {}".format(test_locs))

        train_locs = utilities.get_locations(train_locs)
        test_locs = utilities.get_locations(test_locs)

        data_in = self.data.get_data()
        label_encoder, pipeline = utilities.prepare_skl_interface(data_in, self.classifier)

        train_data = data_in[data_in["Environment"].isin(train_locs)]
        test_data  = data_in[data_in["Environment"].isin(test_locs)]

        accuracy, conf_mat = utilities.train_and_test(train_data, test_data, pipeline, label_encoder, self.srp_dict,  save_cls=save_classifier, out_folder=self.output_folder)
    
        all_metrics = {"overall_accuracy" : (accuracy, 0), 
                       "per_class_accuracy": (metrics.getPCaccuracy(conf_mat), np.zeros(4)), 
                       "per_class_precision": (metrics.getPCPrecision(conf_mat), np.zeros(4)), 
                       "per_class_recall": (metrics.getPCRecall(conf_mat), np.zeros(4)), 
                       "per_class_iou": (metrics.getPCIoU(conf_mat), np.zeros(4))}
        
        metrics.print_metrics(all_metrics, self.paper_metrics_only)


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
    To run cross validation experiment: 
        
        python classificationExpts.py --run_cross_val --locs_list DAB DA DB
    
    To run generalization experiment:

        python classificationExpts.py --run_gen --train_locs_list SA --test_locs_list SB --save_cls

    To train and save a classifier on a data subset:

        python classificationExpts.py --train_save_cls --locs_list SAB --split_ratio 0.15 --save_cls

    """
    parser = ExtractorArgsParser(description='Run classification experiments on the acoustic_data', usage=usage)

    parser.add_argument('--input',  dest='input',  default=None, help='Path to extracted samples')
    parser.add_argument('--output', dest='output', default=None, help='Output folder to store the saved classifier.')

    parser.add_argument('--extract_feats', dest='extract_feats', action='store_true',
                        help='If specified, features are extracted rather than read from a csv file stored in ./config folder')
    parser.add_argument('--save_feats', dest='save_feats', action='store_true', 
                        help='If specified, features are extracted are saved into ./config folder')

    parser.add_argument('--run_cross_val', action='store_true', help='Runs the cross validation experiment. (Table 3 in the paper)')
    parser.add_argument('--locs_list', nargs="+", default=["DAB", "DA", "DB"], help='List of Location IDs to run cross validation / train and test a classifier. To specify multiple subsets, separate the arguments with a space e.g --locs_list SA1 SA2')

    parser.add_argument('--run_gen',  action='store_true', help='Runs the generalization experiment. (Table 4 in the paper)')
    parser.add_argument('--train_locs_list', nargs="+", default=["SA"], help='List of locations in the training set')
    parser.add_argument('--test_locs_list', nargs="+", default=["SB"], help='List of locations in the training set')

    parser.add_argument('--train_save_cls', action='store_true', help='Train a classifier on specified subset and optionally save it.')
    parser.add_argument('--split_ratio', default=0.15, type=float, help='Ratio to split the data into train and test')
    parser.add_argument('--save_cls', action='store_true', help='Save the trained classifier.')

    #hyperparameters
    parser.add_argument('--L',  default=2, type=int, help='Number of segments')
    parser.add_argument('--C',  default=1, type=float, help='Classifier regularization')
    parser.add_argument('--no_data_aug', action='store_true', help='Disable data augmentation')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')


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
    
    print("-- Hyperparameters -- \n")
    print("Seed: {}".format(parsed.seed))
    print("L: {}".format(parsed.L))
    print("C: {}".format(parsed.C))
    print("Data Aug: {}".format(not parsed.no_data_aug))


    # create the data object
    data = utilities.AudioData(L=parsed.L)

    if parsed.extract_feats:
        if parsed.input is None:
            raise ValueError("Please specify path to extracted one second audio samples at the flag --input.")
        else:
            data.extract_data(data_path=parsed.input, save=parsed.save_feats)
    else:
        data.read_csv()

    # initialize the classifier
    classifier = SVC(kernel='linear', C=parsed.C, random_state=parsed.seed, probability=True)
    
    expts = Expts(data, parsed.output, classifier=classifier, random_state=parsed.seed, data_aug=(not parsed.no_data_aug))

    # train and save classifier
    if parsed.train_save_cls:
        expts.run_train_and_save_classifier(locs_in=parsed.locs_list, save_classifier=parsed.save_cls, data_split_ratio=parsed.split_ratio)
    
    # cross validation per location specified
    if parsed.run_cross_val:
        for loc in parsed.locs_list:
            expts.run_cross_validation([loc])
    
    # generalization experiment
    if parsed.run_gen:
        expts.run_generalisation(train_locs=parsed.train_locs_list, test_locs=parsed.test_locs_list, save_classifier=parsed.save_cls)


if __name__ == "__main__":
    main()
