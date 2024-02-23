from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt
from utils.AverageMeter import AverageMeter
from utils.criterion import *
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing as preprocessing
import sys
from math import pi
from math import cos
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim



def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.031, alpha=0.01, num_iter=10, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


#adv_x = pgd(model, data, target, epsilon=0.03, num_steps=20, step_size=0.007, random_start=False)
def pgd(model,
        X,
        y,
        epsilon=8 / 255,
        num_steps=20,
        step_size=0.01,
        random_start=True):
    out = model(X)
    is_correct_natural = (out.max(1)[1] == y).float().cpu().numpy()
    perturbation = torch.zeros_like(X, requires_grad=True)

    if random_start:
        perturbation = torch.rand_like(X, requires_grad=True)
        perturbation.data = perturbation.data * 2 * epsilon - epsilon

    is_correct_adv = []
    opt = optim.SGD([perturbation], lr=1e-3)  # This is just to clear the grad

    for _ in range(num_steps):
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X + perturbation), y)

        loss.backward()

        perturbation.data = (
            perturbation + step_size * perturbation.grad.detach().sign()).clamp(
            -epsilon, epsilon)
        perturbation.data = torch.min(torch.max(perturbation.detach(), -X),
                                      1 - X)  # clip X+delta to [0,1]
        X_pgd = Variable(torch.clamp(X.data + perturbation.data, 0, 1.0),
                         requires_grad=False)
        
    return X_pgd


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
####################################################################################
####################### TRAINING LOSSSES ###############################
##############################################################################

def loss_soft_reg_ep(preds, labels, soft_labels, device, args):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes

    L_c = -torch.mean(torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1))   # Soft labels
    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    

    loss = L_c + args.reg1 * L_p + args.reg2 * L_e


    # prob_one_hot = torch.zeros_like(prob)
    # prob_one_hot.scatter_(1, torch.argmax(prob, dim=1).unsqueeze(1), 1)

    # #prob = torch.argmax(prob, dim=1)  ##### changed


    return prob, loss

##############################################################################
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def loss_mixup_reg_ep(preds, labels, targets_a, targets_b, device, lam, args):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes

    mixup_loss_a = -torch.mean(torch.sum(targets_a * F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(targets_b * F.log_softmax(preds, dim=1), dim=1))
    mixup_loss = lam * mixup_loss_a + (1 - lam) * mixup_loss_b         ###mixup_loss

    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    loss = mixup_loss + args.reg1 * L_p + args.reg2 * L_e
    return prob, loss


##############################################################################

def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, unlabeled_indexes, prev_results=None ):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print(len(train_loader.dataset))

    # switch to train mode
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []

    end = time.time()

    results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)

    if args.loss_term == "Reg_ep":
        print("Training with cross entropy and regularization for soft labels and for predicting different classes (Reg_ep)")
    elif args.loss_term == "MixUp_ep":
        print("Training with Mixup and regularization for soft labels and for predicting different classes (MixUp_ep)")
        alpha = args.Mixup_Alpha
        print("Mixup alpha value:{}".format(alpha))

    # if torch.cuda.device_count() > 1:
    #   model = nn.DataParallel(model)
    #   print("Using", torch.cuda.device_count(), "GPUs!")
  
    # Move the model to the device
    model = model.to(device)    

    counter = 1
    for imgs, img_pslab, labels, soft_labels, index in train_loader:
        #print("len of index is ", len(index))
        images, labels, soft_labels = imgs.to(device), labels.to(device), soft_labels.to(device)

        if args.DApseudolab == "False":
            images_pslab = img_pslab.to(device)
  

        if args.loss_term == "MixUp_ep":
            if args.dropout > 0.0 and args.drop_extra_forward == "True":
                if args.network == "PreactResNet18_WNdrop":
                    tempdrop = model.drop
                    model.drop = 0.0

                elif args.network == "WRN28_5_wn" or args.network == "resnet18_wndrop":
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            tempdrop = m.p
                            m.p = 0.0
                else:
                    tempdrop = model.drop.p
                    model.drop.p = 0.0

            if args.DApseudolab == "False":
                optimizer.zero_grad()
                
                #delta = pgd_linf(model, images_pslab, labels, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False)
                """
                delta = trades_loss(model=model,
                           x_natural=images_pslab,
                           y=labels,
                           optimizer=optimizer,
                           step_size=0.003,
                           epsilon=0.031,
                           perturb_steps=10,
                           beta=6.0)
                """
                output_x1 = model(images_pslab)
                output_x1.detach_()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                output_x1 = model(images)
                output_x1.detach_()
                optimizer.zero_grad()

            if args.dropout > 0.0 and args.drop_extra_forward == "True":
                if args.network == "PreactResNet18_WNdrop":
                    model.drop = tempdrop

                elif args.network == "WRN28_5_wn" or args.network == "resnet18_wndrop":
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            m.p = tempdrop
                else:
                    model.drop.p = tempdrop

            images1, targets_a, targets_b, lam = mixup_data(images, soft_labels, alpha, device)

        #fgsm attack
        
        loss_trades = trades_loss(model=model,
                           x_natural=images,
                           y=labels,
                           optimizer=optimizer,
                           step_size=0.003,
                           epsilon=0.031,
                           perturb_steps=10,
                           beta=6.0)
        #delta = pgd_linf(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False)
        

        # outputs = model(images1)
        
        # compute output
        #outputs = model(images)

        if args.loss_term == "Reg_ep":
            outputs = model(images)
            prob, loss_reg = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)

        elif args.loss_term == "MixUp_ep":
            outputs = model(images1)
            prob = F.softmax(output_x1, dim=1)
            # prob = torch.zeros_like(probs)
            # prob.scatter_(1, torch.argmax(prob, dim=1).unsqueeze(1), 1)

            prob_mixup, loss_reg = loss_mixup_reg_ep(outputs, labels, targets_a, targets_b, device, lam, args)
            outputs = output_x1

        
        results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist() 
        

        
        
        loss=loss_reg + loss_trades
        
        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 1])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                       prec1, optimizer.param_groups[0]['lr']))
        counter = counter + 1

    if args.swa == 'True':
        if epoch > args.swa_start and epoch%args.swa_freq == 0 :
            optimizer.update_swa()        

    # update soft labels
            
    if args.dataset_type == 'ssl':
      train_loader.dataset.update_labels(results, unlabeled_indexes)  #,prev_results

    else:
        train_loader.dataset.update_labels(results, unlabeled_indexes)


    # prev_results.append(results)

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum,results

###################################################################################

def testing(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            adv_x = pgd(model, data, target, epsilon=0.03, num_steps=20, step_size=0.007, random_start=False)
            
            output = model(adv_x)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)

"""
def testing(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            delta = torch.zeros_like(data, requires_grad=True)
            output=model(data+delta)
            output = F.log_softmax(output, dim=1)
            #loss = nn.CrossEntropyLoss()(model(data + delta.detach()), target)
            loss = F.nll_loss(output,target, reduction='sum')
            loss.backward(retain_graph=True)
            delta_grad=delta.grad.detach()
            delta = 0.1 * delta_grad.sign()
            output = model(data+delta)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))
            delta_grad.zero_()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)
"""

def validating(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, _, target, _, _, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)


