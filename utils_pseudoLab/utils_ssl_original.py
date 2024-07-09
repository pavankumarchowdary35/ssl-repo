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
import math



class PGD_Linf():

    def __init__(self, model, epsilon=8*4/255, step_size=4/255, num_steps=10, random_start=True, target_mode= False, criterion= 'kl', bn_mode='eval', train=False, vat=False):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.vat = vat

    def perturb(self, x_nat, targets=None):
        print(x_nat)
        if self.bn_mode == 'eval':
            self.model.eval()
            
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            #self.model.zero_grad()
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                loss.backward()
                grad = x_adv.grad
            elif self.criterion == "kl":
                if self.vat:
                    loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat).detach(), dim = 1))
                    grad = torch.autograd.grad(loss, [x_adv])[0]
                else:
                    loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat), dim = 1))
                    grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "revkl":
                loss = self.criterion_kl(F.log_softmax(self.model(x_nat), dim=1), F.softmax(outputs, dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "js":
                nat_probs = F.softmax(self.model(x_nat), dim=1)
                adv_probs = F.softmax(outputs, dim=1)
                mean_probs = (nat_probs + adv_probs)/2
                loss =  (self.criterion_kl(mean_probs.log(), nat_probs) + self.criterion_kl(mean_probs.log(), adv_probs))/2
                grad = torch.autograd.grad(loss, [x_adv])[0]
            if self.target_mode:
                x_adv = x_adv - self.step_size * grad.sign()
            else:
                x_adv = x_adv + self.step_size * grad.sign()
            
            x_adv = torch.min(torch.max(x_adv, x_nat - self.epsilon), x_nat + self.epsilon)
            x_adv = torch.clamp(x_adv, min=0, max=1).detach()
            d_adv = torch.clamp(x_adv - x_nat, min=-self.epsilon, max=self.epsilon).detach()
            
        if self.train:
            self.model.train()
        
        
        return x_adv, d_adv





def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.031, alpha=0.01, num_iter=10, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
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



def adversarial_loss(model, x_natural,y, optimizer, attack):
    # model.eval()  # Set the model to evaluation mode
    delta = attack(model, x_natural, y)
    # print(delta)
    yp = model(x_natural+delta)# Generate adversarial examples
    yp = F.softmax(yp, dim=1)

    loss = nn.CrossEntropyLoss()(yp, y)
    # print("loss is ", loss)

    return loss


def adversarial_loss_old(model, x_natural,y, optimizer, attack):
    # model.eval()  # Set the model to evaluation mode
    x_adv = attack(model, X =x_natural, y=y)
    # print(delta)
    yp = model(x_adv)# Generate adversarial examples
    yp = F.softmax(yp, dim=1)

    loss = nn.CrossEntropyLoss()(yp, y)
    # print("loss is ", loss)

    return loss
    
    
    
def adversarial_loss_new(model,x_natural,y,optimizer):
    pgd_attack = PGD_Linf(model=model, epsilon=8*4/255, step_size=4/255, num_steps=10, random_start=True, criterion='kl', bn_mode='eval', train=False, vat=False)
    adv_images, _ = pgd_attack.perturb(x_nat=x_natural, targets=y)
    outputs = model(adv_images)
    yp = F.log_softmax(outputs, dim=1)
    loss = nn.CrossEntropyLoss()(yp, y)
    return loss
      

#adv_x = pgd(model, data, target, epsilon=0.03, num_steps=20, step_size=0.007, random_start=False)
def pgd(model,
        X,
        y,
        epsilon=8 / 255,
        num_steps=10,
        step_size=0.01,
        random_start=False):
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
    criterion_kl = nn.KLDivLoss(size_average=False)
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



# def trades_loss(model,
#                 x_natural,
#                 y,
#                 optimizer,
#                 epoch,
#                 step_size=0.003,
#                 epsilon=0.031,
#                 perturb_steps=10,
#                 beta=1.0,
#                 rampup_epochs=50, # Number of epochs for the ramp-up
#                 distance='l_inf'):
#     criterion_kl = nn.KLDivLoss(reduction='sum')
#     model.eval()
#     batch_size = len(x_natural)
#     x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
#     if distance == 'l_inf':
#         for _ in range(perturb_steps):
#             x_adv.requires_grad_()
#             with torch.enable_grad():
#                 loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
#                                        F.softmax(model(x_natural), dim=1))
#             grad = torch.autograd.grad(loss_kl, [x_adv])[0]
#             x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#             x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#             x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     elif distance == 'l_2':
#         delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
#         delta = Variable(delta.data, requires_grad=True)

#         # Setup optimizers
#         optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

#         for _ in range(perturb_steps):
#             adv = x_natural + delta

#             # optimize
#             optimizer_delta.zero_grad()
#             with torch.enable_grad():
#                 loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
#                                            F.softmax(model(x_natural), dim=1))
#             loss.backward()
#             # renorming gradient
#             grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
#             delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
#             # avoid nan or inf if gradient is 0
#             if (grad_norms == 0).any():
#                 delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
#             optimizer_delta.step()

#             # projection
#             delta.data.add_(x_natural)
#             delta.data.clamp_(0, 1).sub_(x_natural)
#             delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
#         x_adv = Variable(x_natural + delta, requires_grad=False)
#     else:
#         x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     model.train()

#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     # zero gradient
#     optimizer.zero_grad()
#     # calculate robust loss
#     logits = model(x_natural)
#     loss_natural = F.cross_entropy(logits, y)
    
#     # Calculate ramp-up factor based on epochs
#     ramp_value = min(1.0, max(0.0, float(epoch) / rampup_epochs))
#     # Adjust beta using the ramp-up factor
#     adjusted_beta = beta * ramp_value

#     loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
#                                                     F.softmax(model(x_natural), dim=1))
#     loss = loss_natural + adjusted_beta * loss_robust
#     return loss





def warmup_loss(model,
                x_natural,
                y,
                optimizer):
    
    model.train()
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss = loss_natural
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

def train_CrossEntropy(args, model,model_teacher, device, train_loader, optimizer, epoch, unlabeled_indexes, prev_results=None):
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
    results_teacher = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)

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

            images, targets_a, targets_b, lam = mixup_data(images, soft_labels, alpha, device)

        #fgsm attack
        
        if args.dataset_type != "ssl_warmUp":
            loss_trades = trades_loss(model=model,
                               x_natural=images,
                               y=labels,
                               optimizer=optimizer,
                               # epoch=epoch,
                               step_size=0.003,
                               epsilon=0.031,
                               perturb_steps=10,
                               beta=1,
                               # rampup_epochs=50,
                               distance='l_inf')
            
            # adv_loss = adversarial_loss(model=model, x_natural= images,y =labels, optimizer=optimizer, attack= pgd_linf)  
            # adv_loss = adversarial_loss(model=model, x_natural= images,y=labels, optimizer=optimizer, attack=pgd_linf)   
            adv_loss = adversarial_loss_old(model=model, x_natural= images,y=labels, optimizer=optimizer, attack=pgd)
            
        if args.dataset_type == "ssl_warmUp":
            warmup_train_loss = warmup_loss(model=model,
                x_natural = images,
                y=labels,
                optimizer=optimizer)
        #delta = pgd_linf(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False) 
        

        # outputs = model(images1)
        
        # compute output
        outputs = model(images)

        if args.loss_term == "Reg_ep":
            
            prob, loss_reg = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)

        elif args.loss_term == "MixUp_ep":
            
            prob = F.softmax(output_x1, dim=1)
            # prob = torch.zeros_like(probs)
            # prob.scatter_(1, torch.argmax(prob, dim=1).unsqueeze(1), 1)

            prob_mixup, loss_reg = loss_mixup_reg_ep(outputs, labels, targets_a, targets_b, device, lam, args)
            outputs = output_x1


#         if epoch == 1:
#             # On the first epoch, make predictions with model_teacher
#             print('came inside the loop')
#             with torch.no_grad():
#                 checkpoint = torch.load('checkpoint_paper/best.pth.tar')
#                 model_teacher.load_state_dict(checkpoint['state_dict'])
#                 # model_teacher.load_state_dict(torch.load('wrn-28-5_algo-fixmatch_lrsche-Cosine_numlabels-4000_seed-0/best.pth.tar'))
#                 model_teacher.eval()
#                 model_teacher.to(device)
#                 if args.DApseudolab == "False":
#                     images_pslab = img_pslab.to(device)
#                     outputs_new = model_teacher(images_pslab)
#                 else:
#                     images = imgs.to(device)
#                     outputs_new = model_teacher(images)
#                 prob_new = torch.softmax(outputs_new, dim=1)

#             results_teacher[index.detach().numpy().tolist()] = prob_new.cpu().detach().numpy().tolist()
#             print('results_teacher was updated with teacher predictions')
        
        results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist() 
                
        if args.dataset_type != "ssl_warmUp":
            loss=loss_reg + loss_trades + adv_loss
        else:
            loss=loss_reg + warmup_train_loss 
            
        
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
            swa_optimizer.update_swa()
            
        if epoch >= args.swa_start:
            swa_optimizer.bn_update(train_loader, model, device)    

    if args.swa == 'True':
        if epoch > args.swa_start and epoch%args.swa_freq == 0 :
            optimizer.update_swa()        

    # update soft labels
    # if epoch == 1:
    #     if args.dataset_type == 'ssl':
    #         train_loader.dataset.update_labels(results, unlabeled_indexes)  #,prev_results
    #     else:
    #         train_loader.dataset.update_labels(results, unlabeled_indexes)      ### if the training is warmup, then the unlabeled indexes will ne zero. so even if we update it is not a problem
    
    
    # train_loader.dataset.update_labels(results, unlabeled_indexes)
    # prev_results.append(results)

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum,results

###################################################################################


def _pgd_whitebox(model,                         ###### from trades loss paper
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=10,
                  step_size=0.003):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def testing(args,model, device, test_loader):            ### from trades loss paper
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)

    
    
# def testing(args, model, device, test_loader):                                ### from cvpr paper
#     model.eval()
#     loss_per_batch = []
#     correct = 0
#     total_samples = 0

#     # Instantiate the PGD_Linf attack object
#     pgd_attack = PGD_Linf(model=model, epsilon=0.03, step_size=0.007, num_steps=10, random_start=False)

    
#     for batch_idx, (data, target) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)

#         # Generate adversarial examples using PGD_Linf attack
#         adv_x, _ = pgd_attack.perturb(data, targets=target)

#         # Forward pass with adversarial examples
#         output = model(adv_x)
#         output = F.log_softmax(output, dim=1)

#         # Compute loss
#         loss = F.nll_loss(output, target, reduction='sum').item()
#         loss_per_batch.append(loss)

#         # Calculate accuracy
#         pred = output.max(1, keepdim=True)[1]
#         correct += pred.eq(target.view_as(pred)).sum().item()
#         total_samples += len(data)

#     # Compute average loss and accuracy
#     test_loss = np.sum(loss_per_batch) / total_samples
#     accuracy = 100. * correct / total_samples

#     # Print results
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#         test_loss, correct, total_samples, accuracy))

#     return ([np.mean(loss_per_batch)], [accuracy])    
    



def testing(args, model, device, test_loader):                                  #### from pgd function
    model.eval()
    loss_per_batch = []
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            adv_x = pgd(model, data, target, epsilon=0.03, num_steps=10, step_size=0.007, random_start=False)
            
            output = model(adv_x)
            output = F.softmax(output, dim=1)
            loss = F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(loss)
            
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)

    test_loss = np.sum(loss_per_batch) / total_samples
    accuracy = 100. * correct / total_samples

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, total_samples, accuracy))

    return ([np.mean(loss_per_batch)], [accuracy])





# def testing(args, model, device, test_loader, epsilon, num_steps, step_size):               ### from trades
#     model.eval()
#     loss_per_batch = []
#     correct = 0
#     total_samples = 0
    
#     criterion_kl = nn.KLDivLoss(reduction='sum')
    
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.to(device), target.to(device)
#             x_natural = data.clone().detach().requires_grad_(True)
#             y = target
            
#             # Generate adversarial example
#             x_adv = x_natural + 0.001 * torch.randn_like(x_natural).cuda().detach()  # Add noise
            
#             for _ in range(num_steps):
#                 x_adv.requires_grad_()
#                 with torch.enable_grad():
#                     loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
#                                            F.softmax(model(x_natural), dim=1))
#                 grad = torch.autograd.grad(loss_kl, [x_adv])[0]
#                 x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#                 x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#                 x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
#             x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)  # Convert to variable
            
#             output = model(x_adv)
#             output = F.log_softmax(output, dim=1)
#             loss = F.nll_loss(output, target, reduction='sum').item()
#             loss_per_batch.append(loss)
            
#             pred = output.max(1, keepdim=True)[1]  # Get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             total_samples += len(data)

#     test_loss = np.sum(loss_per_batch) / total_samples
#     accuracy = 100. * correct / total_samples

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#         test_loss, correct, total_samples, accuracy))

#     return ([np.mean(loss_per_batch)], [accuracy])


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


