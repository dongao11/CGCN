import numpy as np
from utils import io_utils
from data import generator
from torch.autograd import Variable
import sklearn
from sklearn import metrics


def test_one_shot(args, model, test_samples=5000, partition='test',thebestoa=0,thebestaa=0,thebestkappa=0,thebesteveryoa=0):
    io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')

    io.cprint('\n**** TESTING WITH %s ***' % (partition,))

    loader = generator.Generator(args.dataset_root, args, partition=partition, dataset=args.dataset)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples/args.batch_size_test)
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

    acc = np.zeros([1])
    A = np.zeros([CLASS_NUM])
    k = np.zeros([1])
    best_predict_all = []
    best_acc_all = 0.0
    best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None
    class_every=[0]*CLASS_NUM
    class_total=[0]*CLASS_NUM
    classeveryaa=[0]*CLASS_NUM
    oaclasseveryoa=[0]*CLASS_NUM

    theAAacc=0
    theOAacc=0
    theOAacce=0
    kappa=0
    p0=0
    pe=0
    for i in range(iterations):
        data = loader.get_task_batch(batch_size=args.batch_size_test, n_way=args.test_N_way,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra)
        [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, hidden_labels,labels_spec,_] = data

        if args.cuda:
            xi_s = [batch_xi.cuda() for batch_xi in xi_s]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            hidden_labels = hidden_labels.cuda()
            x = x.cuda()
        else:
            labels_yi = labels_yi_cpu

        xi_s = [Variable(batch_xi) for batch_xi in xi_s]
        labels_yi = [Variable(label_yi) for label_yi in labels_yi]
        oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
        hidden_labels = Variable(hidden_labels)
        x = Variable(x)

        # Compute embedding from x and xi_s
        z = [enc_nn(batch_zxi)[-1] for batch_zxi in x]
        zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

        # Compute metric from embeddings
        output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
        output = out_logits
        output = output.reshape(args.batch_size_test * args.query_set, -1)

        y_pred = softmax_module.forward(output)
        y_pred = y_pred.data.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)


        labels_spec=labels_spec.transpose(1, 0)
        labels_spec = labels_spec.reshape(args.batch_size_test * args.query_set, -1)
        labels_spec=labels_spec.numpy()
        labels_spec_x=np.zeros(y_pred.shape[0])


        labels_x_cpu = labels_x_cpu.transpose(1, 0)
        labels_x_cpu = labels_x_cpu.reshape(args.batch_size_test * args.query_set, -1)
        labels_x_cpu = labels_x_cpu.numpy()
        labels_x_cpu = np.argmax(labels_x_cpu, axis=1)

        for row_i in range(y_pred.shape[0]):
            labels_spec_x[row_i] = labels_spec[row_i][labels_x_cpu[row_i]]  # 对应batchsize里面的batch要预测哪个类
            if y_pred[row_i] == labels_x_cpu[row_i]:
                class_every[(labels_spec_x[row_i]).astype('int')]+=1
            class_total[(labels_spec_x[row_i]).astype('int')]+=1



        for row_i in range(y_pred.shape[0]):
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1



        if (i+1) % 100 == 0:
            io.cprint('{} correct from {} \tthe OA Accuracy is : {:.3f}%)'.format(correct, total, 100.0*correct/total))
            theOAacce=correct/total
            p0 = theOAacce
            theAAacc = 0
            pe = 0
            for ii in range(CLASS_NUM):
                theAAacc = (class_every[ii] / class_total[ii]) + theAAacc
                pe = ((class_every[ii]) * (class_total[ii])) + pe
                classeveryaa[ii] = 100.0 * class_every[ii] / class_total[ii]
                if partition == "test":
                    thebesteveryoa[ii] = classeveryaa[ii]
            theAAacc = (theAAacc / CLASS_NUM) * 100
            pe = pe / (total * total)
            kappa = (p0 - pe) / (1 - pe)
            if partition=="test":
                if theOAacce>thebestoa:
                    thebestoa=theOAacce
                    thebestaa = theAAacc
                    thebestkappa = kappa
                    io.cprint('the AA Accuracy is : {:.3f}%)'.format(theAAacc))
                    io.cprint('the kappa Accuracy is : {:.3f}%)'.format(kappa * 100))



    io.cprint('{} correct OA Accuracy from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))
    theOAacce = correct / total
    p0 = theOAacce
    theAAacc = 0
    pe = 0
    for ii in range(CLASS_NUM):
        theAAacc = (class_every[ii] / class_total[ii]) + theAAacc
        pe = ((class_every[ii]) * (class_total[ii])) + pe
    theAAacc = (theAAacc / CLASS_NUM) * 100
    pe = pe / (total * total)
    kappa = (p0 - pe) / (1 - pe)
    if partition=="test":
        if theOAacce > thebestoa:
            thebestoa = theOAacce
            thebestaa = theAAacc
            thebestkappa = kappa
            io.cprint('the AA Accuracy is : {:.3f}%)'.format(theAAacc))
            io.cprint('the kappa Accuracy is : {:.3f}%)'.format(kappa * 100))
            for ii in range(CLASS_NUM):
                io.cprint('the class of {} is{} correct from {} \tAccuracy: {:.3f}%)'.format(ii, class_every[ii],
                                                                                             class_total[ii],
                                                                                             100.0 * class_every[ii] /
                                                                                             class_total[ii]))
                classeveryaa[ii] = 100.0 * class_every[ii] / class_total[ii]
                if partition == "test":
                    thebesteveryoa[ii] = classeveryaa[ii]

    io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))
    enc_nn.train()
    metric_nn.train()

    return 100.0 * correct / total,thebestoa,thebestaa,thebestkappa,thebesteveryoa
