import os
import warnings
import metrics
import pickle
import json

import numpy as np
import xml.etree.ElementTree as ET
import pyroomacoustics as pra
import pandas as pd
import scipy.io.wavfile as wavf
import sklearn as skl

from scipy import signal
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import pdb

class AudioData:

    def __init__(self, res=30, freq_range=[50,1500], nfft=2*256, L=2, mic_array=None):

        # set parameters for feature extraction
        self.mic_array = mic_array

        if mic_array is None:
            self.mic_array = loadMicarray()

        self.resolution = res
        self.freq_range = freq_range 
        self.nfft = nfft
        self.L = L

        self.data = None
        self.data_df_save_path = './config/extracted_features.csv'

    def extract_data(self, data_path, save=False):
        
        label_data = pd.read_csv(os.path.join(*[data_path, 'SampleLog.csv']))

        #shifting columns around for easier accessibility later
        label_data = label_data[ ["Class"] + [ col for col in label_data.columns if col != "Class" ] ]
        extracted_data = None

        num_samples = dict.fromkeys(["front", "left", "none", "right"], 0)

        for idx, row in tqdm(label_data.iterrows(), desc='Extracting features: ', total=label_data.shape[0]):

            sample_rate, data = wavf.read(os.path.join(*[data_path, row["Class"], row["ID"]+'.wav']))
            feature = extractSRPFeature(data, sample_rate, self.mic_array, self.resolution, self.freq_range, self.nfft, self.L)

            if extracted_data is None:
                # first set number of columns according to the feature shape
                data_columns = ['feat' + str(x) for x in range(feature.shape[0])]

                #then append the rest of the columns from the label_data
                data_columns.extend(label_data.columns)

                # now create the dataframe and add the feature and label details
                extracted_data = pd.DataFrame(columns=data_columns)
                extracted_data = extracted_data.append(pd.DataFrame([np.concatenate((feature, row))], columns=data_columns), ignore_index=True)
            else:
                extracted_data = extracted_data.append(pd.DataFrame([np.concatenate((feature, row))], columns=data_columns), ignore_index=True)
            
            num_samples[row["Class"]] += 1

        self.data = extracted_data

        if save:
            extracted_data.to_csv(self.data_df_save_path, index=False)

    def get_data(self):
        return self.data

    def read_csv(self, csv_path=None):
        if csv_path is None:
            self.data = pd.read_csv(self.data_df_save_path)
        else:
            print(" --- Please ensure csv is of the format given by the file: {} --- ".format(self.data_df_save_path))
            self.data = pd.read_csv(csv_path)


def get_locations(locs_in=["SAB"]):

    temp_locs_in = locs_in.copy()

    loc_ids = {"type_A": ["A1", "A2"],  
               "type_B": ["B1", "B2", "B3"]}

    for loc in temp_locs_in:
        if loc == "SAB" or loc == "DAB":
            temp_ids = loc_ids["type_A"] + loc_ids["type_B"]
            locs_in.remove(loc)
            locs_in.extend([loc[0] + id for id in temp_ids])

        elif loc == "DA" or loc == "SA":
            temp_ids = loc_ids["type_A"]
            locs_in.remove(loc)
            locs_in.extend([loc[0] + id for id in temp_ids])
            
        elif loc == "DB" or loc == "SB":
            temp_ids = loc_ids["type_B"]
            locs_in.remove(loc)
            locs_in.extend([loc[0] + id for id in temp_ids])

    return list(set(locs_in))


def prepare_skl_interface(data_in, classifier):

    #prepare the sklearn interface
    le = skl.preprocessing.LabelEncoder()
    le.fit(data_in["Class"].unique())
    scaler = skl.preprocessing.StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', classifier)])
    
    return le, pipeline


def partitionPanda(vector, fold, k):
    size = vector.shape[0]
    start = (size//k)*fold
    end = (size//k)*(fold+1)
    validation = vector.iloc[start:end,:]

    indices = range(start, end)
    mask = np.ones(vector.shape[0], dtype=bool)
    mask[indices] = False
    training = vector.iloc[mask,:]

    return training, validation


def cross_validation(pipeline, n_folds, data, le, srp_dict, data_aug=True):

    un_classes = data["Class"].unique()   # get classes
    
    # initialize variables that hold metrics
    per_class_acc = []             # Per class accuracy
    per_class_prec = []            # Per class precision
    per_class_rec = []             # Per class recall
    per_class_iou = []             # Per class IoU
    validation_folds_score = []    # Overall accuracy on Validation folds 
    CO = None                      # Confusion matrix summed over folds           


    # iterate over the folds
    for fold in range(0, n_folds):
        training_set = pd.DataFrame(columns=data.columns)
        validation_set = pd.DataFrame(columns=data.columns)

        # exception for the loo case
        if n_folds == data.shape[0]:
            training_set, validation_set = partitionPanda(data, fold, n_folds)
        else:
            # otherwise make sure that classes are equally distributed
            for single_label in un_classes:
                df_sl = data[data["Class"] == single_label]
                df_sl = df_sl.reset_index(drop=True)
                train_snippet, validation_snippet = partitionPanda(df_sl, fold, n_folds)
                training_set = training_set.append(train_snippet, ignore_index=True)
                validation_set = validation_set.append(validation_snippet, ignore_index=True)

        # train classifier and get predictions on validation
        accuracy, C = train_and_test(training_set, validation_set, pipeline, le, srp_dict, data_aug=data_aug)

        # aggregate the metrics
        validation_folds_score.append(accuracy)

        if CO is None:
            CO = C
        else:
            CO = CO + C

        per_class_acc.extend([metrics.getPCaccuracy(C)])
        per_class_prec.extend([metrics.getPCPrecision(C)])
        per_class_rec.extend([metrics.getPCRecall(C)])
        per_class_iou.extend([metrics.getPCIoU(C)])

    metrics_dict = {"overall_accuracy" : (np.mean(validation_folds_score), np.std(validation_folds_score)), 
                    "per_class_accuracy": (np.mean(per_class_acc, axis=0), np.std(per_class_acc, axis=0)), 
                    "per_class_precision": (np.mean(per_class_prec, axis=0), np.std(per_class_prec, axis=0)), 
                    "per_class_recall": (np.mean(per_class_rec, axis=0), np.std(per_class_rec, axis=0)), 
                    "per_class_iou": (np.mean(per_class_iou, axis=0), np.std(per_class_iou, axis=0))}


    return metrics_dict


def do_data_augmentation(data_in, res, nsegs):

    # prepare output and and cut out left and right class samples
    columns = data_in.columns
    right = data_in[data_in["Class"] == 'right']
    left = data_in[data_in["Class"] == 'left']
    df_right = pd.DataFrame(columns=columns)
    df_left = pd.DataFrame(columns=columns)
    data_out = pd.DataFrame(columns=columns)

    # first deal with left samples
    for index, rows in right.iterrows():
        np_con = None

        for i in range(nsegs):
            if np_con is None:
                np_con = np.flip(rows[0:res], 0).to_numpy()
            else:
                np_con = np.concatenate((np_con, np.flip(rows[i*res:(i+1)*res].to_numpy(), 0)))
        df_right = df_right.append(pd.DataFrame([np.concatenate((np_con, np.array(['left']), rows[nsegs*res+1:len(rows)]))], columns=columns), ignore_index=True)
        #(appended data format)                np.concatenate((flipped array, class label direction flipped, other details))

    for index, rows in left.iterrows():
        np_con = np.empty(0)
        np_con = None

        # depending on feature parameters nsegs and res do the flipping 
        for i in range(nsegs):
            if np_con is None:
                np_con = np.flip(rows[0:res], 0).to_numpy()
            else:
                np_con = np.concatenate((np_con, np.flip(rows[i*res:(i+1)*res], 0).to_numpy()))   
                np_con = np.concatenate((np_con, np.flip(rows[i*res:(i+1)*res], 0).to_numpy()))   
                np_con = np.concatenate((np_con, np.flip(rows[i*res:(i+1)*res], 0).to_numpy()))   
        df_left = df_left.append(pd.DataFrame([np.concatenate((np_con, np.array(['right']), rows[nsegs*res+1:len(rows)]))], columns=columns), ignore_index=True)
        
    # create output data and return
    data_out = data_out.append(data_in, ignore_index=True)
    data_out = data_out.append(df_left, ignore_index=True)
    data_out = data_out.append(df_right, ignore_index=True)

    return data_out


def train_and_test(train_set, test_set, pipeline, le, srp_dict=None, save_cls=False, out_folder=None, data_aug=True):

    # do flip based data augmentation
    if data_aug:
        if srp_dict is not None:
            train_set = do_data_augmentation(train_set, srp_dict['res'], srp_dict['nsegs'])
    
    # check until which column features are stored
    i_max = 1
    for i, col in enumerate(train_set.columns):
        if 'feat' in col:
            i_max = i + 1
    
    # split the dataframe to get features and append the transformed labels
    data_train = np.split(train_set.to_numpy(), [i_max], axis=1)
    data_train[1] = le.transform(train_set["Class"])  

    data_test = np.split(test_set.to_numpy(), [i_max], axis=1)
    data_test[1] = le.transform(test_set["Class"])

    # fit the classifier and predict on the test set
    pipeline.fit(data_train[0], data_train[1])
    test_predicted = pipeline.predict(data_test[0])

    accuracy_score = skl.metrics.accuracy_score(data_test[1], test_predicted)

    # extract confusion matrix and metrics
    conf_mat = skl.metrics.confusion_matrix(data_test[1], test_predicted, labels=le.transform(le.classes_))

    if save_cls:
        if out_folder is None:
            save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_classifier')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join(out_folder, 'saved_classifier/')
            os.makedirs(save_dir, exist_ok=True)

        print("Saving Classifier to {} ... ".format(save_dir))

        locs_in_train = train_set["Environment"].unique()
        save_string = "_".join(locs_in_train)

        pickle.dump((pipeline), open(os.path.join(*[save_dir, save_string + '_classifier.obj']), "wb"))
        test_set = test_set.drop_duplicates(subset=["Recording ID"])
        test_set["ID"].to_csv(os.path.join(*[save_dir, save_string + '_test_bags.csv']), index=False, header=True)

    return accuracy_score, conf_mat


def makeDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loadMicarray():
    ar_x = []
    ar_y = []
    
    # iterrate through the xml to get all locations
    root = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/config/ourmicarray_56.xml').getroot()
    for type_tag in root.findall('pos'):
        ar_x.append(type_tag.get('x'))
        ar_y.append(type_tag.get('y'))

    # set up the array vector
    micArray = np.zeros([len(ar_x), 3])
    micArray[:,1] = ar_x
    micArray[:,2] = ar_y

    micArrayConfig = """
  _______________________________________________________________
   Loading microphone Array with {} microphones.  
                                            -O  |
                                -O              |
                    -O                          |
        -O               |Z                     |            ┌ ┐
                         |    _Y            -O  |            |X|
                         |___/  -O              | micArray = |Y|
                    -O    \                     |            |Z|
        -O                 \X                   |            └ ┘
                                            -O  |
                                -O              |
                    -O                          |
        -O                                      | 
  _______________________________________________________________\n\n
        """.format(micArray.shape[0])

    print(micArrayConfig)

    return micArray


def extractSRPFeature(dataIn, sampleRate, micArray, resolution, freqRange=[10,1200], nfft=2*256, L=2):
    # generate fft lengths and filter mics and create doa algorithm
    doaProcessor = pra.doa.algorithms['SRP'](micArray.transpose(), sampleRate, nfft, azimuth=np.linspace(-90.,90., resolution)*np.pi/180, max_four=4)
    
    # extract the stft from parameters
    container = []
    for i in range(dataIn.shape[1]):
        _, _, stft = signal.stft(dataIn[:,i], sampleRate, nperseg=nfft)
        container.append(stft)
    container = np.stack(container)
    
    # split the stft into L segments
    segments = []
    delta_t = container.shape[-1] // L 
    for i in range(L):
        segments.append(container[:, :, i*delta_t:(i+1)*delta_t])
    # pdb.set_trace()
    # container = [container[:, :, 0:94], container[:, :, 94:94+94]]

    # apply the doa algorithm for each specified segment according to parameters
    feature = []
    for i in range(L):
        doaProcessor.locate_sources(segments[i], freq_range=freqRange)
        feature.append(doaProcessor.grid.values)

    return np.concatenate(feature)


def detectionFolder(folder, score_threshold=0, height_threshold=0):
    """
    score_threshold : range 0-1.0
    """
    detection_fpath = os.path.join(folder, "camera_baseline_detections.json")
    if not os.path.isfile(detection_fpath):
        raise ValueError("No file {} found.")

    with open("{}/camera_baseline_detections.json".format(folder), 'r') as f:
        detection_summary = json.load(f)
    detections_per_frame = detection_summary['detections_per_frame']

    filtered_detections_per_frame = []
    for detections in detections_per_frame:
        filter_detections = [(box, score)
                             for box, score, class_str
                             in zip(detections['boxes'], detections['scores'],
                                    detections['classes_str'])
                             if ((class_str == 'car' or class_str == 'motorcycle')
                                 and score >= score_threshold
                                 and np.abs(box[1]-box[3]) > height_threshold)]
        filtered_detections_per_frame.append(filter_detections)

    return filtered_detections_per_frame


