import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

    def Precision(self):
        Precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis = 0)
        return Precision

    def Recall(self):
        Recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis = 1)
        return Recall

    def F1(self):
        Precision = self.Precision()
        Recall = self.Recall()
        F1 = 2 * Precision * Recall / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
        return np.array([0.0, OA])  # Return array format consistent with other metrics

    def Kappa(self):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = np.sum(self.confusion_matrix)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(self.confusion_matrix) / float(sum_total)
        kappa = (po - pe) / (1 - pe)
        return np.array([0.0, kappa])  # Return array format consistent with other metrics

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = gt_image >= 0
        label = gt_image[mask].astype('int')
        pred = pre_image[mask].astype('int')
        n_class = self.num_class
        confusion_matrix = np.zeros((n_class, n_class))
        confusion_matrix = confusion_matrix.astype('int64')
        for i in range(len(label)):
            if label[i] < n_class and pred[i] < n_class:
                confusion_matrix[label[i], pred[i]] += 1
            else:
                print('number of classes should be equal to one of the label. label_i:%d, pred_i:%d' % (label[i] ,pred[i]))
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
