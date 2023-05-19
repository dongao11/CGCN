from __future__ import print_function

from typing import List

import torch.utils.data as data
import torch
import numpy as np
import random
import os

from numpy import ndarray
from torch.autograd import Variable

import utils2
from . import UP
from . import IP
from . import salinas
from . import PC
from . import Houston





class Generator(data.Dataset):
    def __init__(self, root, args, partition='train', dataset='UP'):
        self.root = root
        self.partition = partition
        self.args = args
        utils2.same_seeds(0)
        pwd = os.getcwd()
        #
        f_pwd = os.path.abspath(os.path.dirname(pwd) )


        if dataset=="IP":
            test_data = f_pwd + '/few-shot-gnn-master/datasets/IP/indian_pines_corrected.mat'
            test_label = f_pwd + '/few-shot-gnn-master/datasets/IP/indian_pines_gt.mat'
        elif dataset=="UP":
            test_data=f_pwd+"/few-shot-gnn-master/datasets/paviaU/paviaU.mat"
            test_label=f_pwd+'/few-shot-gnn-master/datasets/paviaU/paviaU_gt.mat'
        elif dataset == "salinas":
            test_data = f_pwd + "/few-shot-gnn-master/datasets/salinas/salinas_corrected.mat"
            test_label = f_pwd + '/few-shot-gnn-master/datasets/salinas/salinas_gt.mat'
        elif dataset == "PC":
            test_data = f_pwd + "/few-shot-gnn-master/datasets/pavia/pavia.mat"
            test_label = f_pwd + '/few-shot-gnn-master/datasets/pavia/pavia_gt.mat'
        elif dataset == "Houston":
            test_data = f_pwd + "/few-shot-gnn-master/datasets/Houston/Houston.mat"
            test_label = f_pwd + '/few-shot-gnn-master/datasets/Houston/Houston_gt.mat'

        Data_Band_Scaler, GroundTruth = utils2.load_data(test_data, test_label)

        assert (dataset == 'UP' or
                dataset == 'IP' or dataset=='salinas' or 'PC'or'Houston'), 'Incorrect dataset partition'
        self.dataset = dataset

        if self.dataset == 'UP':
            self.input_channels = 103
            self.size = (9, 9)

        elif self.dataset=='IP':
            self.input_channels = 200
            self.size = (9, 9)#m

        elif self.dataset=='salinas':
            self.input_channels = 204
            self.size = (9, 9)#m

        elif self.dataset=='PC':
            self.input_channels = 102
            self.size = (9, 9)#m

        elif self.dataset=='Houston':
            self.input_channels = 144
            self.size = (9, 9)#m

        if dataset == 'UP':
            self.data, self.label_encoder, self.test_loader, self.best_G, self.best_RandPerm, self.best_Row, self.best_Column = UP.get_train_test_loader(
                Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,
                partition=self.partition)
        elif dataset == 'IP':
            self.data, self.label_encoder, self.test_loader, self.best_G, self.best_RandPerm, self.best_Row, self.best_Column = IP.get_train_test_loader(
                Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, partition=self.partition)

        elif dataset == 'salinas':
            self.data, self.label_encoder, self.test_loader, self.best_G, self.best_RandPerm, self.best_Row, self.best_Column = salinas.get_train_test_loader(
                Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, partition=self.partition)

        elif dataset == 'PC':
            self.data, self.label_encoder, self.test_loader, self.best_G, self.best_RandPerm, self.best_Row, self.best_Column = PC.get_train_test_loader(
                Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, partition=self.partition)


        elif dataset == 'Houston':

            self.data, self.label_encoder, self.test_loader, self.best_G, self.best_RandPerm, self.best_Row, self.best_Column = Houston.get_train_test_loader(

                Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, partition=self.partition)

        else:
            raise NotImplementedError

        self.class_encoder = {}

    def rotate_image(self, image, times):
        rotated_image = np.zeros(image.shape)#
        for channel in range(image.shape[0]):
            rotated_image[channel, :, :] = np.rot90(image[channel, :, :], k=times)#

        return rotated_image

    def get_task_batch(self, batch_size=5, n_way=20, num_shots=1, unlabeled_extra=0, cuda=False, variable=False):

        batch_x = []
        labels_x = []
        labels_spec = []
        labels_x_global = []
        target_distances = []
        hidden_labels = []
        numeric_labels = []
        batches_xi: list[ndarray]
        batches_xi, labels_yi, oracles_yi = [], [], []
        cross_labels = []
        querysets = self.args.query_set
        # Init variables
        hidden_labels=np.zeros((batch_size, n_way * num_shots + 1), dtype='float32')
        target_distances=np.zeros((batch_size, n_way * num_shots), dtype='float32')
        for i in range(querysets):
            batch_x.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]),
                                       dtype='float32'))
            labels_x.append(np.zeros((batch_size, n_way), dtype='float32'))
            labels_spec.append(np.zeros((batch_size, n_way), dtype='float32'))

            labels_x_global.append(np.zeros(batch_size, dtype='int64'))





        for i in range(n_way * num_shots):
            batches_xi.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]),dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            cross_labels.append(np.zeros((batch_size), dtype='float32'))
            # Iterate over tasks for the same batch

        for batch_counter in range(batch_size):

            positive_class = [0] * querysets
            samplecountera=[0]*n_way
            for i in range(querysets):
                positive_class[i] = random.randint(0, n_way - 1)
                samplecountera[positive_class[i]]=samplecountera[positive_class[i]]+1

            # Sample random classes for this TASK
            classes_ = list(self.data.keys())
            sampled_classes = random.sample(classes_,n_way)
            indexes_perm = np.random.permutation(n_way * num_shots)

            counter = 0
            # querysample = []
            # supportsample = []

            for class_counter, class_ in enumerate(sampled_classes):


                for iq in range(querysets):
                    if class_counter == positive_class[iq]:

                        samples = random.sample(self.data[class_], 1)




                        batch_x[iq][batch_counter, :, :, :] = samples[0]
                        labels_x[iq][batch_counter, class_counter] = 1
                        labels_spec[iq][batch_counter, class_counter] = class_



                samples = random.sample(self.data[class_], num_shots)
                sample_label = []
                for sli in range(0, len(samples)):
                    sample_label.append(class_)


                for s_i in range(0, len(samples)):

                    batches_xi[indexes_perm[counter]][batch_counter, :, :, :] = samples[s_i]
                    cross_labels[indexes_perm[counter]][batch_counter] = sample_label[s_i]
                    if s_i < unlabeled_extra:

                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 0
                        hidden_labels[batch_counter, indexes_perm[counter] + 1] = 1
                    else:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 1

                    oracles_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    target_distances[batch_counter, indexes_perm[counter]] = 0

                    counter += 1

            numeric_labels.append(positive_class)


        batch_x=np.array(batch_x)
        labels_x=np.array(labels_x)

        labels_spec=np.array(labels_spec)

        labels_x_global=np.array(labels_x_global)



        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        cross_labels = torch.tensor(cross_labels)
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)



        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi,
                      torch.from_numpy(hidden_labels), torch.from_numpy(labels_spec), cross_labels]


        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def cast_variable(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_variable(input[i])
        else:
            return Variable(input)

        return input

