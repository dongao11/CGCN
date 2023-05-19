from __future__ import print_function
import os
import argparse
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data import generator
from utils import io_utils
import models.models as models
import test
import numpy as np

parser = argparse.ArgumentParser(description='Few-Shot Learning with Graph Neural Networks')
parser.add_argument('--exp_name', type=str, default='h1649way', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_test', type=int, default=10, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--decay_interval', type=int, default=10000, metavar='N',
                    help='Learning rate decay interval')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=300000, metavar='N',
                    help='how many batches between each model saving')
parser.add_argument('--test_interval', type=int, default=1000, metavar='N',
                    help='how many batches between each test')
parser.add_argument('--test_N_way', type=int, default=15, metavar='N',
                    help='Number of classes for doing each classification run')
parser.add_argument('--train_N_way', type=int, default=15, metavar='N',
                    help='Number of classes for doing each training comparison')
parser.add_argument('--test_N_shots', type=int, default=1, metavar='N',
                    help='Number of shots in test')
parser.add_argument('--train_N_shots', type=int, default=1, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--query_set', type=int, default=3, metavar='N',
                    help='Number of query sets')

parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--metric_network', type=str, default='gnn_iclr_nl', metavar='N',
                    help='gnn_iclr_nl' + 'gnn_iclr_active')
parser.add_argument('--active_random', type=int, default=0, metavar='N',
                    help='random active ? ')
parser.add_argument('--dataset_root', type=str, default='datasets', metavar='N',
                    help='Root dataset')
parser.add_argument('--test_samples', type=int, default=30000, metavar='N',
                    help='Number of shots')
parser.add_argument('--dataset', type=str, default='Houston', metavar='N',
                    help='UP  IP')
parser.add_argument('--dec_lr', type=int, default=2000, metavar='N',
                    help='Decreasing the learning rate every x iterations')
args = parser.parse_args()

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp models/models.py checkpoints' + '/' + args.exp_name + '/' + 'models.py.backup')
_init_()

io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')
io.cprint(str(args))

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    io.cprint('Using GPU : ' + str(torch.cuda.current_device())+' from '+str(torch.cuda.device_count())+' devices')
    torch.cuda.manual_seed(args.seed)
else:
    io.cprint('Using CPU')


def train_batch(model, data):

    [enc_nn, metric_nn, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi, oracles_yi, hidden_labels,cross_labels] = data


    z = [enc_nn(batch_xxi)[-1] for batch_xxi in batch_x]#

    zi_s = [enc_nn(batch_xi)[-1] for batch_xi in batches_xi]#
    ztzi_s=z+zi_s
    tensor_zi_s=torch.stack(zi_s)#
    tensor_ztzi_s=torch.stack(ztzi_s)#
    tensor_ztzi_s=tensor_ztzi_s.transpose(1,0)#
    tensor_zi_s=tensor_zi_s.transpose(1,0)#


    # Compute metric from embeddings
    crossobj, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])#
    out_logits=out_logits.reshape(args.batch_size*args.query_set,-1)
    logsoft_prob = softmax_module.forward(out_logits)






    label_x=label_x.transpose(1,0)
    label_x_og_numpy=label_x.cpu().data.numpy()
    label_x_og_sc=np.argmax(label_x_og_numpy,axis=2)
    label_x=label_x.reshape(args.batch_size*args.query_set,-1)
    label_x_numpy = label_x.cpu().data.numpy()
    cross_labels_numpy=cross_labels.cpu().data.numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
    cross_labels_obj=Variable(torch.LongTensor(cross_labels_numpy))
    cross_labels_obj=cross_labels_obj.transpose(1,0)
    label_x_og_sc=Variable(torch.LongTensor(label_x_og_sc))


    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
        cross_labels_obj=cross_labels_obj.cuda()
        label_x_og_sc=label_x_og_sc.cuda()

    loss = F.nll_loss(logsoft_prob, formatted_label_x)
    loss_of_contrative = instance_contrastive_Loss()
    loss2 = loss_of_contrative(crossobj, cross_labels_obj)
    loss3=loss_of_contrative(tensor_zi_s,cross_labels_obj)
    a1=0.9
    a2=0.2
    loss = a1 * loss+ a2 * loss2+0.1 * loss3
    loss.backward()

    return loss


def train():
    train_loader = generator.Generator(args.dataset_root, args, partition='train', dataset=args.dataset)
    io.cprint('Batch size: '+str(args.batch_size))


    enc_nn = models.load_model('enc_nn', args, io)
    metric_nn = models.load_model('metric_nn', args, io)

    if enc_nn is None or metric_nn is None:
        enc_nn, metric_nn = models.create_models(args=args)
    softmax_module = models.SoftmaxModule()

    if args.cuda:
        enc_nn.cuda()
        metric_nn.cuda()

    io.cprint(str(enc_nn))
    io.cprint(str(metric_nn))

    weight_decay = 0
    if args.dataset == 'IP':
        print('Weight decay '+str(1e-3))
        weight_decay = 1e-3
    elif args.dataset=='UP':
        print('Weight decay ' + str(1e-3))
        weight_decay = 1e-3
    elif args.dataset=='PC':
        print('Weight decay ' + str(1e-3))
        weight_decay = 1e-3
    elif args.dataset == 'salinas':
        print('Weight decay ' + str(1e-3))
        weight_decay = 1e-3
    elif args.dataset == 'Houston':
        print('Weight decay ' + str(1e-3))
        weight_decay = 1e-3
    opt_enc_nn = optim.Adam(enc_nn.parameters(), lr=args.lr, weight_decay=weight_decay)
    opt_metric_nn = optim.Adam(metric_nn.parameters(), lr=args.lr, weight_decay=weight_decay)

    if args.dataset=="IP":
        CLASS_NUM=16
    elif args.dataset=="PC":
        CLASS_NUM=9
    elif args.dataset=="salinas":
        CLASS_NUM=16
    elif args.dataset == "UP":
        CLASS_NUM = 9
    else:
        CLASS_NUM = 15
    enc_nn.train()
    metric_nn.train()
    counter = 0
    total_loss = 0
    val_acc, val_acc_aux = 0, 0
    thebestoa = 0
    thebestaa = 0
    thebestkappa = 0
    thebesteveryoa = [0] * CLASS_NUM
    thebestoam = 0
    thebestaam = 0
    thebestkappam = 0
    thebesteveryoam = [0] * CLASS_NUM
    test_acc = 0
    for batch_idx in range(args.iterations+20):

        ####################
        # Train
        ####################
        data = train_loader.get_task_batch(batch_size=args.batch_size, n_way=args.train_N_way,
                                           unlabeled_extra=args.unlabeled_extra, num_shots=args.train_N_shots,
                                           cuda=args.cuda, variable=True)
        [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi, hidden_labels,_,cross_labels] = data


        opt_enc_nn.zero_grad()
        opt_metric_nn.zero_grad()

        loss_d_metric = train_batch(model=[enc_nn, metric_nn, softmax_module],
                                    data=[batch_x, label_x, batches_xi, labels_yi, oracles_yi, hidden_labels,cross_labels])

        opt_enc_nn.step()
        opt_metric_nn.step()


        adjust_learning_rate(optimizers=[opt_enc_nn, opt_metric_nn], lr=args.lr, iter=batch_idx)

        ####################
        # Display
        ####################
        counter += 1
        total_loss += loss_d_metric.item()
        if batch_idx % args.log_interval == 0:
                display_str = 'Train Iter: {}'.format(batch_idx)
                display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss/counter)
                io.cprint(display_str)
                counter = 0
                total_loss = 0

        ####################
        # Test
        ####################
        if (batch_idx + 1) % (args.test_interval+20) == 0 or batch_idx == 20:
            if batch_idx == 20:
                test_samples = 100
            else:
                test_samples = 3000
            if args.dataset == 'IP':
                val_acc_aux,_,_,_,_ = test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                                 test_samples=test_samples*5, partition='val',thebestoa=thebestoa,thebestaa=thebestaa,thebestkappa=thebestkappa,thebesteveryoa=thebesteveryoa)
            if args.dataset=='UP':
                val_acc_aux,_,_,_,_ = test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                                     test_samples=test_samples * 5, partition='val',thebestoa=thebestoa,thebestaa=thebestaa,thebestkappa=thebestkappa,thebesteveryoa=thebesteveryoa)
            if args.dataset=='salinas':
                val_acc_aux,_,_,_,_ = test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                                     test_samples=test_samples * 5, partition='val',thebestoa=thebestoa,thebestaa=thebestaa,thebestkappa=thebestkappa,thebesteveryoa=thebesteveryoa)
            if args.dataset=='PC':
                val_acc_aux,_,_,_,_ = test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                                     test_samples=test_samples * 5, partition='val',thebestoa=thebestoa,thebestaa=thebestaa,thebestkappa=thebestkappa,thebesteveryoa=thebesteveryoa)
            if args.dataset=='Houston':
                val_acc_aux,_,_,_,_ = test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                                     test_samples=test_samples * 5, partition='val',thebestoa=thebestoa,thebestaa=thebestaa,thebestkappa=thebestkappa,thebesteveryoa=thebesteveryoa)

            test_acc_aux,thebestoa,thebestaa,thebestkappa,thebesteveryoa = test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                              test_samples=test_samples*5, partition='test',thebestoa=thebestoa,thebestaa=thebestaa,thebestkappa=thebestkappa,thebesteveryoa=thebesteveryoa)
            test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                               test_samples=test_samples, partition='train',thebestoa=thebestoa,thebestaa=thebestaa,thebestkappa=thebestkappa,thebesteveryoa=thebesteveryoa)

            enc_nn.train()
            metric_nn.train()

            if val_acc_aux is not None and val_acc_aux >= val_acc:
                test_acc = test_acc_aux
                val_acc = val_acc_aux
                thebestaam=thebestaa
                thebestkappam=thebestkappa
                for ii in range(CLASS_NUM):
                   thebesteveryoam[ii] = thebesteveryoa[ii]



            io.cprint("Best test accuracy {:.4f} \n".format(test_acc))
            io.cprint('the best OA Accuracy is {:.3f}%)'.format(thebestoa * 100))
            io.cprint('the best AA Accuracy is {:.3f}%)'.format(thebestaam))
            io.cprint('the best kappa Accuracy is {:.3f}%)'.format(thebestkappam * 100))
            for ii in range(CLASS_NUM):
                io.cprint('the class of {} best Accuracy is : {:.3f}%)'.format((ii + 1), thebesteveryoa[ii]))

        ####################
        # Save model
        ####################
        if (batch_idx + 1) % args.save_interval == 0:
            torch.save(enc_nn, 'checkpoints/%s/models/enc_nn.t7' % args.exp_name)
            torch.save(metric_nn, 'checkpoints/%s/models/metric_nn.t7' % args.exp_name)

    # Test after training
    test.test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                       test_samples=args.test_samples,thebestoa=thebestoa,thebestaa=thebestaa,thebestkappa=thebestkappa,thebesteveryoa=thebesteveryoa)




def adjust_learning_rate(optimizers, lr, iter):
    new_lr = lr * (0.5**(int(iter/args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr







class instance_contrastive_Loss(torch.nn.Module):
    def __init__(self):
        super(instance_contrastive_Loss, self).__init__()

    def forward(self, out_1, label, temperature=0.5):
        shot = args.train_N_shots
        #
        out_1 = F.normalize(out_1, dim=-1)
        #
        sim_matrix = torch.exp(
            torch.bmm(out_1, out_1.transpose(2, 1).contiguous()) / temperature)  # .transpose(example89, dim0=1, dim1=2)

        #
        diag = torch.eye(out_1.shape[1],device=sim_matrix.device)
        diag = diag.unsqueeze(0)
        mask = (torch.ones_like(sim_matrix) - diag).bool()

        #
        negative_sim = sim_matrix.masked_select(mask).view(out_1.shape[0], out_1.shape[1], -1)

        negative_sim = negative_sim.sum(dim=-1)

        #

        mask = label.unsqueeze(1)

        mask = mask.repeat(1, out_1.shape[1], 1)

        mask2 = mask.transpose(2, 1)
        mask = mask2 - mask

        mask[mask != 0] = -1  #
        mask[mask == 0] = 1  #
        mask[mask < 0] = 0  #
        diag = torch.eye(out_1.shape[1],device=sim_matrix.device)
        diag = diag.unsqueeze(0)

        mask = (mask - diag).bool()

        pos_sim = sim_matrix.masked_select(mask).view(out_1.shape[0], out_1.shape[1], -1).mean(dim=-1)

        if shot == 1:
            return (- torch.log(1 / negative_sim)).mean()
        else:
            return (- torch.log(pos_sim / negative_sim)).mean()

if __name__ == "__main__":
    train()

