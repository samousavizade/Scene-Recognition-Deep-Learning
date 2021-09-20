import os
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets
import torchvision.transforms as transforms
import copy
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


class TrainStatistics:
    def __init__(self, best_model_validation_accuracy, accuracy_tracks, loss_tracks):
        self.best_model_validation_accuracy = best_model_validation_accuracy
        self.accuracy_tracks = accuracy_tracks
        self.loss_tracks = loss_tracks

    def get_best_epoch_statistics_dict(self):
        best_index = 0
        best_top1_accuracy = 0
        for index in range(len(self.accuracy_tracks)):
            top1_accuracy, top5_accuracy = self.accuracy_tracks[VALIDATION_STR][index]

            if top1_accuracy > best_top1_accuracy:
                best_top1_accuracy = top1_accuracy
                best_index = index

        train_loss = self.loss_tracks[TRAIN_STR][best_index]
        validation_loss = self.loss_tracks[VALIDATION_STR][best_index]

        train_top1, train_top5 = self.accuracy_tracks[TRAIN_STR][best_index]
        validation_top1, validation_top5 = self.accuracy_tracks[VALIDATION_STR][best_index]

        return {
            f'{PART_STR}/' + 'Loss/': train_loss,
            f'{PART_STR}/' + 'Loss/': validation_loss,
            f'{PART_STR}/' + 'Accuracy/': train_top1,
            f'{PART_STR}/' + 'Accuracy/': train_top5,
            f'{PART_STR}/' + 'Accuracy/': validation_top1,
            f'{PART_STR}/' + 'Accuracy/': validation_top5,
        }


PART_STR = 'Part5'
TRAIN_STR = 'Train'
VALIDATION_STR = 'Test'
SIZE_STR = 'size'
BATCH_SIZE_STR = 'batch_size'
LEARNING_RATE_STR = 'lr'
ACCURACY_STR = 'Accuracy'
LOSS_STR = 'Loss'
EPOCH_NUMBER_STR = 'epoch_number'
WEIGHT_DECAY_STR = 'weight_decay'

summery_writer = SummaryWriter(f'runs/{PART_STR}')


class HyperParameterConfigs:

    def __init__(self, epochs_number, image_data_size, learning_rate, batch_size, weight_decay):
        self.epochs_number = epochs_number
        self.image_data_size = image_data_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    def to_dict(self):
        return {
            'learning rate': self.learning_rate,
            'batch size': self.batch_size,
            'weight decay factor': self.weight_decay,
            'epochs number': self.epochs_number
        }

    def __str__(self):
        return f'Epochs Number : {self.epochs_number}, ' \
               f'Image Size : ({self.image_data_size},{self.image_data_size}), ' \
               f'Learning Rate : {self.learning_rate}, ' \
               f'Batch Size : {self.batch_size}, ' \
               f'Weight Decay: {self.weight_decay}'


class NeuralNetworkTrainer:

    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.best_model, self.best_configs, self.best_accuracy = None, None, 0

    @staticmethod
    def top_k_corrects(predictions, trues, device, topk=(1, 5)):
        with torch.no_grad():
            M = max(topk)
            _, class_predict = predictions.topk(k=M, dim=1)
            class_predict = class_predict.t()
            target_reshaped = trues.view(1, -1).expand_as(class_predict)
            correct = (class_predict == target_reshaped)
            topk_list = torch.zeros(len(topk), dtype=torch.long, device=device)
            for index, k in enumerate(topk):
                indices = correct[:k]
                flattened_indices = indices.reshape(-1).float()
                topk_correct = flattened_indices.sum(dim=0, keepdim=True)
                topk_list[index] = topk_correct

            return tuple(topk_list)

    def train_net_with_hyper_parameters(self, configs: HyperParameterConfigs):

        train_directory = f'{self.data_directory}/{TRAIN_STR}/'
        validation_directory = f'{self.data_directory}/{VALIDATION_STR}/'

        phases = [TRAIN_STR, VALIDATION_STR]

        print('Hyper Parameters : \n')
        print(str(configs))

        size = (configs.image_data_size, configs.image_data_size)
        lr = configs.learning_rate
        batch_size = int(configs.batch_size)
        epochs_number = int(configs.epochs_number)
        weight_decay = configs.weight_decay

        probability = .5
        mean, standard_deviation = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_preprocess_transforms = transforms.Compose([
            transforms.Resize(size=size),
            transforms.RandomHorizontalFlip(p=probability),
            transforms.RandomPerspective(p=probability, distortion_scale=.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation)
        ])

        train_dataset = torchvision.datasets.ImageFolder(train_directory, transform=train_preprocess_transforms)

        validation_preprocess_transforms = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation)
        ])

        validation_dataset = torchvision.datasets.ImageFolder(validation_directory,
                                                              transform=validation_preprocess_transforms)

        datasets = {TRAIN_STR: train_dataset,
                    VALIDATION_STR: validation_dataset}

        shuffle = True
        drop_last = True

        num_workers = 4
        train_dataloader = data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           drop_last=drop_last,
                                           num_workers=num_workers,
                                           pin_memory=True)

        validation_dataloader = data.DataLoader(validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                drop_last=drop_last,
                                                num_workers=num_workers,
                                                pin_memory=True)

        data_loaders = {TRAIN_STR: train_dataloader,
                        VALIDATION_STR: validation_dataloader}

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'device is : {device}')
        num_classes = len(train_dataset.classes)

        alex_net: models.AlexNet = models.alexnet(pretrained=True, progress=True)

        last_classifier_layer_input_features_number = alex_net.classifier[6].in_features

        fully_connected_layer = nn.Linear(last_classifier_layer_input_features_number, num_classes)
        alex_net.classifier._modules['6'] = fully_connected_layer

        images, labels = next(iter(train_dataloader))
        grid = torchvision.utils.make_grid(images)
        summery_writer.add_image("images", grid)
        summery_writer.add_graph(alex_net, images)
        summery_writer.flush()
        summery_writer.close()

        alex_net = alex_net.to(device)

        optimizer = optim.Adam(params=alex_net.parameters(), lr=lr, weight_decay=weight_decay)

        loss_function = nn.CrossEntropyLoss()

        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.82)

        # train model
        fine_tuned_trained_net, train_statistics = self.train_model(alex_net,
                                                                    data_loaders, datasets,
                                                                    loss_function,
                                                                    optimizer,
                                                                    lr_scheduler,
                                                                    device,
                                                                    phases,
                                                                    epochs_number)

        return fine_tuned_trained_net, train_statistics

    def hyper_parameter_tune(self, hyper_parameters_space_dict):
        hps_dict = hyper_parameters_space_dict
        best_configs = None
        best_model = None
        best_accuracy = 0

        for image_data_size in hps_dict[SIZE_STR]:
            for batch_size in hps_dict[BATCH_SIZE_STR]:
                for learning_rate in hps_dict[LEARNING_RATE_STR]:
                    for epochs_number in hps_dict[EPOCH_NUMBER_STR]:
                        for weight_decay in hps_dict[WEIGHT_DECAY_STR]:
                            current_configs = HyperParameterConfigs(epochs_number,
                                                                    image_data_size,
                                                                    learning_rate,
                                                                    batch_size,
                                                                    weight_decay)

                            current_model, statistics = self.train_net_with_hyper_parameters(current_configs)

                            statistics: TrainStatistics

                            if statistics.best_model_validation_accuracy > best_accuracy:
                                best_configs = current_configs
                                best_model = current_model
                                best_accuracy = statistics.best_model_validation_accuracy

                            # send to tensorboard accuracy / loss diagram
                            for i in range(epochs_number):
                                epoch_train_phase_top1_accuracy, epoch_train_phase_top5_accuracy = \
                                    statistics.accuracy_tracks[TRAIN_STR][i]
                                epoch_validation_phase_top1_accuracy, epoch_validation_phase_top5_accuracy = \
                                    statistics.accuracy_tracks[VALIDATION_STR][i]

                                summery_writer.add_scalars(f'{PART_STR}/' + 'Accuracy',
                                                           {
                                                               f'{TRAIN_STR} Top 1': epoch_train_phase_top1_accuracy,
                                                               f'{TRAIN_STR} Top 5': epoch_train_phase_top5_accuracy,
                                                               f'{VALIDATION_STR} Top 1': epoch_validation_phase_top1_accuracy,
                                                               f'{VALIDATION_STR} Top 5': epoch_validation_phase_top5_accuracy,
                                                           },
                                                           i)

                                summery_writer.add_scalars(f'{PART_STR}/' + 'Loss',
                                                           {
                                                               TRAIN_STR: statistics.loss_tracks[TRAIN_STR][i],
                                                               VALIDATION_STR: statistics.loss_tracks[VALIDATION_STR][i]
                                                           },
                                                           i)

        print('Best Configuration is : ', best_configs)
        self.best_model, self.best_configs, self.best_accuracy = best_model, best_configs, best_accuracy

        return self

    def get_best_model(self):
        return self.best_model, self.best_configs, self.best_accuracy

    @staticmethod
    def train_model(model, data_loaders, datasets,
                    loss_function, optimizer, lr_scheduler,
                    device, phases, epochs_number):

        dataset_sizes = {x: len(datasets[x]) for x in phases}

        accuracy_tracks = {TRAIN_STR: [], VALIDATION_STR: []}
        loss_tracks = {TRAIN_STR: [], VALIDATION_STR: []}

        best_model_wts = copy.deepcopy(model.state_dict())
        best_validation_accuracy = torch.tensor([0.0], device=device)

        for epoch in range(epochs_number):
            print('-' * 50)
            print(f'Epoch {epoch + 1}/{epochs_number}')
            print('-' * 50)
            # set to train phase
            model.train()

            train_loss, \
            train_sum_of_top1_corrects, \
            train_sum_of_top5_corrects = torch.tensor(0.0, device=device), \
                                         torch.tensor(0, dtype=torch.long, device=device), \
                                         torch.tensor(0, dtype=torch.long,
                                                      device=device)

            for i, batch in enumerate(data_loaders[TRAIN_STR]):
                inputs, true_labels = batch

                # send to gpu
                inputs, true_labels = inputs.to(device), true_labels.to(device)

                # zero the gradients
                optimizer.zero_grad()

                # forward + backward + optimize

                outputs = model(inputs)
                loss = loss_function(outputs, true_labels)

                loss.backward()
                optimizer.step()

                # print statistics
                train_loss += loss * inputs.size(0)

                top1_corrects, top5_corrects = NeuralNetworkTrainer.top_k_corrects(outputs, true_labels,
                                                                                   device=device)

                train_sum_of_top1_corrects += top1_corrects
                train_sum_of_top5_corrects += top5_corrects

            # calculate train statistics
            epoch_train_phase_loss = train_loss.item() / dataset_sizes[TRAIN_STR]
            epoch_train_phase_top1_accuracy = train_sum_of_top1_corrects.item() / dataset_sizes[TRAIN_STR]
            epoch_train_phase_top5_accuracy = train_sum_of_top5_corrects.item() / dataset_sizes[TRAIN_STR]

            print(f'Train Loss: {epoch_train_phase_loss}')
            print(f'Train Accuracy | Top 1: {epoch_train_phase_top1_accuracy}')
            print(f'Train Accuracy | Top 5: {epoch_train_phase_top5_accuracy}')

            # Validation loss
            validation_loss, validation_sum_of_top1_corrects, validation_sum_of_top5_corrects = torch.tensor(0.0,
                                                                                                             device=device), \
                                                                                                torch.tensor(0,
                                                                                                             device=device), \
                                                                                                torch.tensor(0,
                                                                                                             device=device)

            for i, batch in enumerate(data_loaders[VALIDATION_STR]):
                with torch.no_grad():
                    inputs, true_labels = batch
                    inputs, true_labels = inputs.to(device), true_labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    loss = loss_function(outputs, true_labels)
                    validation_loss += loss * inputs.size(0)

                    top1_corrects, top5_corrects = NeuralNetworkTrainer.top_k_corrects(outputs, true_labels,
                                                                                       device=device)

                    validation_sum_of_top1_corrects += top1_corrects
                    validation_sum_of_top5_corrects += top5_corrects

            # calculate evaluation statistics
            epoch_validation_phase_loss = \
                validation_loss.item() / dataset_sizes[VALIDATION_STR]
            epoch_validation_phase_top1_accuracy = \
                validation_sum_of_top1_corrects.item() / dataset_sizes[VALIDATION_STR]
            epoch_validation_phase_top5_accuracy = \
                validation_sum_of_top5_corrects.item() / dataset_sizes[VALIDATION_STR]

            # send to tensorboard
            summery_writer.add_scalars(f'Current Accuracy:',
                                       {
                                           'Train Top 1': epoch_train_phase_top1_accuracy,
                                           'Train Top 5': epoch_train_phase_top5_accuracy,
                                           'Validation Top 1': epoch_validation_phase_top1_accuracy,
                                           'Validation Top 5': epoch_validation_phase_top5_accuracy,
                                       },
                                       epoch)

            summery_writer.add_scalars(f'Current Loss:',
                                       {
                                           'Train': epoch_train_phase_loss,
                                           'Validation': epoch_validation_phase_loss
                                       },
                                       epoch)

            print(f'Validation Loss: {epoch_validation_phase_loss}')
            print(f'Validation Accuracy | Top 1: {epoch_validation_phase_top1_accuracy}')
            print(f'Validation Accuracy | Top 5: {epoch_validation_phase_top5_accuracy}')

            print()

            lr_scheduler.step()

            # update best model
            if epoch_validation_phase_top1_accuracy > best_validation_accuracy:
                best_validation_accuracy = epoch_validation_phase_top1_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

            # save statistics
            loss_tracks[TRAIN_STR].append(epoch_train_phase_loss)
            loss_tracks[VALIDATION_STR].append(epoch_validation_phase_loss)

            accuracy_tracks[TRAIN_STR].append((epoch_train_phase_top1_accuracy,
                                               epoch_train_phase_top5_accuracy))

            accuracy_tracks[VALIDATION_STR].append((epoch_validation_phase_top1_accuracy,
                                                    epoch_validation_phase_top5_accuracy))

        print('Best Validation Accuracy: {:3f}'.format(best_validation_accuracy))

        # load best model weights
        model.load_state_dict(best_model_wts)

        # save train statistics
        train_statistics = TrainStatistics(best_model_validation_accuracy=best_validation_accuracy,
                                           accuracy_tracks=accuracy_tracks,
                                           loss_tracks=loss_tracks)

        return model, train_statistics


def main():
    data_directory = os.path.abspath('./Data')
    hyper_parameters_space_dict = {
        SIZE_STR: [256],
        BATCH_SIZE_STR: [12],
        LEARNING_RATE_STR: [.3 * 1e-4],
        EPOCH_NUMBER_STR: [45],
        WEIGHT_DECAY_STR: [1e-3]
    }

    model, configs, accuracy = NeuralNetworkTrainer(data_directory). \
        hyper_parameter_tune(hyper_parameters_space_dict). \
        get_best_model()

    print()
    print()
    print('-' * 50)
    print('Model with :')
    print(configs)
    print(f'achieve accuracy: {accuracy}')


if __name__ == '__main__':
    main()
