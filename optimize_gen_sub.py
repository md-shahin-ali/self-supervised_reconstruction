#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import h5py
import numpy.lib.recfunctions as rf
from numpy import linalg as LA
import math
import  os
import pickle


def optimize_generator(dop,G,z,params,train_epoch=1, save_weight= True, gtExists= True):
    with torch.autograd.set_detect_anomaly(True):    
        pathname =  params['filename'].replace('.mat','_'+str(params['gen_base_size'])+'d/weights_GENplusLAT'+str(params['coilEst'])+'_')
        pathname =      pathname+str(params['slice'])+'_'+str(params['nintlPerFrame'])+'arms_'+str(params['siz_l'])+'latVec'+str(params['nFramesDesired'])+'frms_'+str(params['splitRatio'])+'SplitRatio_'+params['splitDist']
        if not(os.path.exists(pathname)):
          os.makedirs(pathname)

        lr_g = params['lr_g']
        lr_z = params['lr_z']
        gpu = params['device']
        batch_sz = params['nBatch']
        legendstring = np.array2string(np.arange(params["siz_l"]))
        legendstring = legendstring[1:-1]
        #optimizer = optim.SGD([
        #{'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_g},
        #{'params': z.z_, 'lr': lr_z}
        #], momentum=(0.9))
        optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_g},
        {'params': z.z_, 'lr': lr_z}
        ], betas=(0.4, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, threshold=0.001, min_lr=[1e-15, 1e-12])

        if isinstance(params['slice'], int):
            nslc = 1
        else:
            nslc = len(params['slice'])

        train_hist = {}
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []
    #----------------------------------------
        valid_loss_min = np.inf
        val_loss_tracker = 0
        G_val_losses = []
    #---------------------------------------
        loss = nn.MSELoss(reduction='sum')

        G_oldi = G.state_dict()
        z_oldi = z.z_.data
        divergence_counter = 0 

        print('training start!')
        start_time = time.time()
        G_losses = []
        nbatches = params['nFramesDesired']//batch_sz  #180
        nindices = nbatches*batch_sz  #180*5=900
    #     ImagesFrmEpochs=[]
    #     LatentVecFromEpochs=[]
        # Epoch loop
        for epoch in range(train_epoch):
            indices=np.arange(0,params['nFramesDesired'])
            random.shuffle(indices)
            #indices = np.random.randint(0,params['nFramesDesired'],params['nFramesDesired'])
            indices = np.reshape(indices[0:nindices],(nbatches,batch_sz))
            epoch_start_time = time.time()
            batch_loss = 0
            # Batch loop
            #-----------
            G.train()
            for batch in range(nbatches):
                G_loss = 0
                optimizer.zero_grad()

                #Slice loop
                #----------

                for slc in range(nslc):
                    G_result = G(z.z_[indices[batch],...,slc])[...,slc]
            #-------------------------------------------------------------------------------
                    if params['ssTrainMode']:
                        G_result_projected = dop.Psub(G_result,indices[batch],slc, mode= 'train')
                    else:
                        G_result_projected = dop.Psub(G_result,indices[batch],slc)
                    if(params["fastMode"]):
                        if params['ssTrainMode']:
                            G_loss += loss(torch.view_as_real(G_result_projected),torch.view_as_real(dop.AtbTrn[indices[batch],...,slc]).to(gpu))
                        else:
                            G_loss += loss(torch.view_as_real(G_result_projected),torch.view_as_real(dop.Atb[indices[batch],...,slc]).to(gpu))
                    else:
                        G_loss += loss(torch.view_as_real(G_result_projected),torch.view_as_real(dop.AtbTrn[indices[batch],...,slc]))

                    if params['ssTrainMode']:
                        G_loss += dop.image_energy_sub(G_result,slc, mode= 'train')  # image regularization to zero out regions outside maks
                    else:
                        G_loss += dop.image_energy_sub(G_result,slc)
                    G_loss += z.KLloss(slc)  # K_L divergence loss per slice 
                #Slice loop end
                #---------------
                G_loss +=  G.weightl1norm()    # Netowrk regularization
                G_loss += z.Reg()     # latent variable regularization  
                
            #-----------------------------------------------------------------------------------


                G_loss.backward()
                #print('before batchloss=%.3f'%batch_loss)
                batch_loss += G_loss.detach()
                optimizer.step()
                #print('After batchloss=%.3f'%batch_loss)
            # Batch loop end
            #---------------
            G_losses.append(batch_loss.item()/(nbatches*batch_sz))
            
        #-----------------------------------------------------------------
            # Validation
            
            # Batch loop
            #-----------
            G.eval()
            with torch.no_grad():
                batch_val_loss = 0
                for batch in range(nbatches):
                    GValLoss = 0
                    #Slice loop
                    #----------
                    for slc in range(nslc):
                        G_result = G(z.z_[indices[batch],...,slc])[...,slc]
                        G_result_projected = dop.Psub(G_result,indices[batch],slc, mode= 'validation').to(gpu)
                        if(params["fastMode"]):
                            GValLoss += loss(torch.view_as_real(G_result_projected),torch.view_as_real(dop.AtbVal[indices[batch],...,slc]).to(gpu))
                        else:
                            GValLoss += loss(torch.view_as_real(G_result_projected),torch.view_as_real(dop.AtbVal[indices[batch],...,slc]))

                        # GValLoss += dop.image_energy_sub(G_result,slc, mode= 'validation')  # image regularization to zero out regions outside maks
                        
                #----------------------------------------------------------------------------------
                    batch_val_loss += GValLoss.detach()
            # Batch loop end
            #---------------
                G_val_losses.append(batch_val_loss.item()/(nbatches*batch_sz))
                if params['ssTrainMode']:
                    scheduler.step(G_val_losses[-1])
                # scheduler.step(batch_val_loss.item())
        #-------------------------------------------------------------------------------------
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            
            #save the best checkpoint
            checkpoint = {
                    "epoch": epoch,
                    "splitRatio": params['splitRatio'],
                    "valid_loss_min":G_val_losses,
                    "AllValLoss": G_val_losses,
                    "AllTrainingLoss": G_losses,
                    "model_state": G.state_dict(),
                    "latVec_state": z.z_,
                    "optim_state": optimizer.state_dict()
                }
            if(epoch >150):
                if G_val_losses[-1] < valid_loss_min:
                    valid_loss_min = G_val_losses[-1]
                    torch.save(checkpoint, os.path.join(pathname,'bestModelValPoint_splitRatio{}_epoch{}_valLoss_{}.pth'.format(params['splitRatio'],epoch, torch.FloatTensor(G_val_losses)[-1]))) 
                    val_loss_tracker = 0 #reset the val loss tracker each time a new lowest val error is achieved  
                else:
                    val_loss_tracker += 1
                # if val_loss_tracker == params['stop_training']:
                #     print(f'Training stopped at {epoch+1}-th')
                #     torch.save(checkpoint, os.path.join(pathname,'ForPlottingLossStats_splitRatio{}_epoch{}_valLoss_{}_trnLoss_{}.pth'.format(params['splitRatio'],epoch, torch.FloatTensor(G_val_losses)[-1], torch.FloatTensor(G_losses)[-1])))
                #     return G,z,train_hist,SER,epoch
            
            # If cost increases, load an old state and decrease step size
#------------------------------------------------------------------------------------------------------
            if(epoch >10):
                if((batch_loss.item() > 1.15*train_hist['G_losses'][-1])): # higher cost
                    G.load_state_dict(G_oldi)
                    z.z_.data = z_oldi
                    print('loading old state; reducing stp siz')
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']*0.98
                    divergence_counter = divergence_counter+1

                else:       # lower cost; converging
                    divergence_counter = divergence_counter-1
                    if((divergence_counter<0)):
                        divergence_counter=0
                    G_oldi = G.state_dict()
                    z_oldi = z.z_.data    
                    train_hist['G_losses'].append(batch_loss.item())
                    path = os.path.join(pathname, 'net_{}_splitRatio{}_5Zreg_epch{}_valLoss_{}_trnLoss_{}.pth'.format('GENplusLAT', params['splitRatio'], epoch, torch.FloatTensor(G_val_losses)[-1], torch.FloatTensor(G_losses)[-1]))
                    if (save_weight):
                        torch.save({'G_oldi':G.state_dict(),'z_oldi':z.z_}, path)
            else:
                G_oldi = G.state_dict()
                z_oldi = z.z_.data 
                train_hist['G_losses'].append(batch_loss.item())
                path = os.path.join(pathname, 'net_{}_splitRatio{}_5Zreg_epch{}_valLoss_{}_trnLoss_{}.pth'.format('GENplusLAT', params['splitRatio'], epoch, torch.FloatTensor(G_val_losses)[-1], torch.FloatTensor(G_losses)[-1]))
                if (save_weight):
                    torch.save({'G_oldi':G.state_dict(),'z_oldi':z.z_}, path)

            # If diverges, exit
            # if(divergence_counter>=1):
            #     print('Optimization diverging; exiting')
            #     return G,z,train_hist,SER,epoch
#------------------------------------------------------------------------------------------------------------------

            #G_oldi = G.state_dict()
            #z_oldi = z.z_.data

            #path = os.path.join(pathname, 'net_{}_Zreg_epch{}_gloss{}.pth'.format('GENplusLAT', epoch,torch.mean(torch.FloatTensor(G_losses))))
            #torch.save({'G_oldi':G.state_dict(),'z_oldi':z.z_}, path)
            # Epoch loop end
            #---------------

            #Display results
            print('[%d/%d] - ptime: %.2f, loss_g: %.3f, val_loss: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.FloatTensor(G_losses)[-1], torch.FloatTensor(G_val_losses)[-1]))

            #Extracting each epoch Generator input and output 
    #         IntermediateGenOutput= G(z.z_[:20,...,-1])
    #         IntermediateLatInput= z.z_[...].data.squeeze().cpu().numpy()
    #         ImagesFrmEpochs.append(IntermediateGenOutput.cpu().data.numpy())
    #         LatentVecFromEpochs.append(IntermediateLatInput)

            # visualization after every 10 epoch
            if(np.mod(epoch,9)==0):
                fig,ax = plt.subplots(nslc,2)   
                if(nslc==1):
                    ax = np.expand_dims(ax,0)
                for sl in range(nslc):
                    G_result = G(z.z_[indices[batch],...,sl])[...,sl]
    #               print(f'the frame indices: {indices[batch]} and gen output shape: {G_result.shape}')

                    test_image1 = G_result[-1].squeeze(0).cpu().data.numpy()

                    ax[sl,0].imshow(abs(test_image1),cmap='gray', vmin=0.0, vmax=.6* np.max(abs(test_image1)))
                    temp = z.z_[...,sl].data.squeeze().cpu().numpy()
                    #               print(f'the latent vector shape is {temp.shape}')
                    ax[sl,1].plot(temp)
                    #ax[sl,1].legend(legendstring,loc='best')          
                plt.pause(0.00001)
                # print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(G_losses))))
        print('Optimization done in %d seconds' %(time.time()-start_time))
        print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.FloatTensor(G_losses)[-1]))
        torch.save(checkpoint, os.path.join(pathname,'ForPlottingLossStats_splitRatio{}_epoch{}_valLoss_{}_trnLoss_{}.pth'.format(params['splitRatio'],epoch, torch.FloatTensor(G_val_losses)[-1], torch.FloatTensor(G_losses)[-1])))
    return G,z,train_hist,epoch
