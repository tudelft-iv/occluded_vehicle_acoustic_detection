import numpy as np

def getPCaccuracy(C):
        pc_accuracies = []
        for i in range(len(C)):
            TP = C[i,i]
            col_sum = np.sum(C[:,i])
            row_sum = np.sum(C[i,:])
            FP = col_sum - TP
            FN = row_sum - TP
            TN = np.sum(C) - TP - FP - FN
            if (TP + TN + FP + FN) == 0:
                pc_accuracies.append(1.0)
            else:
                pc_accuracies.append((TP + TN) / (TP + TN + FP + FN))
        return pc_accuracies

def getPCIoU(C):
        pc_accuracies = []
        for i in range(len(C)):
            TP = C[i,i]
            col_sum = np.sum(C[:,i])
            row_sum = np.sum(C[i,:])
            FP = col_sum - TP
            FN = row_sum - TP
            TN = np.sum(C) - TP - FP - FN
            if (TP + FP + FN) == 0:
                pc_accuracies.append(0.0)
            else:
                pc_accuracies.append((TP) / (TP + FP + FN))
        return pc_accuracies

def getPCRecall(C):
    C = C.astype(float)
    c = np.divide(C.diagonal(), np.sum(C, axis=1), out=np.zeros_like(C.diagonal()), where=np.sum(C, axis=1)!=0)
    return c

def getPCPrecision(C):
    C = C.astype(float)
    c = np.divide(C.diagonal(), np.sum(C, axis=0), out=np.zeros_like(C.diagonal()), where=np.sum(C, axis=1)!=0)
    return c


def print_metrics(metrics_dict, paper_metrics_only=False):
    
    if paper_metrics_only:
        paper_metrics = ["overall_accuracy", "per_class_iou"]
    else:
        paper_metrics = metrics_dict.keys()

    metric_order = ["front", "left", "none", "right"]

    print(" --- Metrics ---\n")
    print("Format: (mean) +/- (std)\n")
    for key, value in metrics_dict.items():
        if key in paper_metrics:
            print(" Metric: {}".format(key))
            
            # for per class metrics, specify the class to which they belong
            if "per_class" in key:
                print(" Values: ")
                for mean, std, clss in zip(value[0], value[1], metric_order):
                    print("\t{}: {} +/- {}".format(clss, mean, std))

            else:     
                print(" Value: {} +/- {}".format(value[0], value[1]))
            
            print(" ==== ")