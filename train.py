import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import scipy.io as sio
from evaluation import eval_aligned_detail, clustering
from calculate_graph import calculate_graphs, calculate_laplacian, calculate_cosine_similarity, knn
from loss import crossview_contrastive_Loss
from stochastic import wasserstein_initialisation, regularise_and_invert
from util import next_batch_aligned, to_tensor, to_numpy

def pretrain(model, optimizer, config, x1_train, x2_train, flag, Y_list, logger, pretrain_path, device):
    for epoch in range(config['training']['pre_epoch']):
        loss_all, loss_rec1, loss_rec2, loss_cl, loss_pre = 0, 0, 0, 0, 0
        for batch_x1_aligned, batch_x2_aligned, batch_x1_mis_aligned, batch_x2_mis_aligned, batch_No in next_batch_aligned(
                x1_train, x2_train, flag, config['training']['batch_size']):
            z1 = model.autoencoder1.encoder(batch_x1_aligned)
            z2 = model.autoencoder2.encoder(batch_x2_aligned)
            z1_mis = model.autoencoder1.encoder(batch_x1_mis_aligned)
            z2_mis = model.autoencoder2.encoder(batch_x2_mis_aligned)
            # Within-view Reconstruction Loss
            recon1 = F.mse_loss(model.autoencoder1.decoder(z1), batch_x1_aligned)
            recon2 = F.mse_loss(model.autoencoder2.decoder(z2), batch_x2_aligned)
            recon3 = F.mse_loss(model.autoencoder1.decoder(z1_mis), batch_x1_mis_aligned)
            recon4 = F.mse_loss(model.autoencoder2.decoder(z2_mis), batch_x2_mis_aligned)
            reconstruction_loss = recon1 + recon2 + recon3 + recon4

            # Cross-view Contrastive_Loss
            cl_loss = crossview_contrastive_Loss(z1, z2, config['training']['alpha'])

            # Cross-view Dual-Prediction Loss
            z1_hat, _ = model.generator1(z1)
            z2_hat, _ = model.generator2(z2)
            pre1 = F.mse_loss(z1_hat, z2, reduction='sum')
            pre2 = F.mse_loss(z2_hat, z1, reduction='sum')
            dualprediction_loss = (pre1 + pre2)

            loss = cl_loss + reconstruction_loss * config['training']['lambda2']

            # we train the autoencoder by L_cl and L_rec first to stabilize
            # the training of the dual prediction
            if epoch >= config['training']['start_dual_prediction']:
                loss += dualprediction_loss * config['training']['lambda1']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            loss_rec1 += recon1.item()
            loss_rec2 += recon2.item()
            loss_pre += dualprediction_loss.item()
            loss_cl += cl_loss.item()

        if (epoch + 1) % config['print_num'] == 0:
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                .format((epoch + 1), config['training']['epoch'], loss_rec1, loss_rec2, loss_pre, loss_cl, loss_all)

            logger.info("\033[2;29m" + output + "\033[0m")

        if (epoch + 1) % config['print_num'] == 0:
            scores = model.evaluation(logger, x1_train, x2_train, Y_list)
    torch.save(model.state_dict(), pretrain_path)

def train(model, optimizer, config, x1_train, x2_train, flag, Y_list, index_mis_aligned, P_gt, logger, device):
    # 指标
    best_acc = 0
    acc, nmi, ari = [], [], []

    # got config
    update = config['training']['update']
    epochs = range(config['training']['epoch'])
    got_init_epoch = config['training']['got']['init_epoch']
    got_update_epoch = config['training']['got']['update_epoch']

    # init got
    fea1, fea2 = get_cat_feature(model, x1_train, x2_train)
    fea1 = to_numpy(fea1)
    fea2 = to_numpy(fea2)
    dim = int(fea1.shape[1] / 2)
    similarity = calculate_cosine_similarity(fea1[~flag], fea2[~flag])
    similarity = to_tensor(similarity, device)
    model.got.init_param(similarity)

    # training
    for epoch in epochs:
        # update P
        if epoch % update == 0:
            fea1, fea2 = get_cat_feature(model, x1_train, x2_train)

            g1, g2, L1_reg, L2_reg = get_got_input(fea1[~flag][:, dim:], fea2[~flag][:, :dim], config, k=config['training']['k'])
            if epoch == 0:
                got_epoch = got_init_epoch
            else:
                got_epoch = got_update_epoch
            P, P_pred = train_got(model.got, L1_reg, L2_reg, optimizer, config, got_epoch, device)
            # eval got delete
            eval_aligned_detail(P, P_pred, index_mis_aligned, Y_list[0])
            P_global = get_global_p(flag, P, device=device)
        x1_recon, x2_recon, z1, z2, z1_hat, z2_hat = model(x1_train, x2_train)
        # Within-view Reconstruction Loss
        recon1 = F.mse_loss(x1_recon, x1_train)
        recon2 = F.mse_loss(x2_recon, x2_train)
        reconstruction_loss = recon1 + recon2

        # Cross-view Contrastive_Loss
        cl_loss = crossview_contrastive_Loss(z1[flag], z2[flag], config['training']['alpha'])

        # Cross-view Dual-Prediction Loss
        pre1 = F.mse_loss(z1_hat, P_global @ z2)
        pre2 = F.mse_loss(P_global @ z2_hat, z1)
        dualprediction_loss = (pre1 + pre2)

        loss = cl_loss + reconstruction_loss * config['training']['lambda2'] + dualprediction_loss * config['training']['lambda1']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % config['print_num'] == 0:
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                .format((epoch + 1), config['training']['epoch'], recon1.item(), recon2.item(), dualprediction_loss.item(), cl_loss.item(), loss.item())

            logger.info("\033[2;29m" + output + "\033[0m")

        if (epoch + 1) % 1 == 0:
            latent_fusion = torch.cat([P_global @ z2_hat.detach(), z1_hat.detach()], dim=1).cpu().numpy()
            # graph1, graph2 = calculate_graphs(fake_z2.detach(), fake_z1.detach())
            # graph2 = P_global @ graph2 @ P_global.t()
            # graph = (graph1 + graph2) * 0.5
            # # scores = clustering([latent_fusion], Y_list[0])
            # scores = post_clustering(graph.cpu().numpy(), Y_list[0], config['cluster_param'])
            scores = clustering([latent_fusion], Y_list[0])
            acc.append(scores['kmeans']['accuracy'])
            nmi.append(scores['kmeans']['NMI'])
            ari.append(scores['kmeans']['ARI'])
            if best_acc < scores['kmeans']['accuracy']:
                best_acc = scores['kmeans']['accuracy']
            logger.info("\033[2;29m" + 'epoch' + str(epoch) + '     ===>view_concat ' + str(scores) + "\033[0m")
    print('best_accuracy: %.4f' % best_acc)
    num = 5
    acc = np.array(acc)
    idx = np.argsort(-acc)
    nmi = np.array(nmi)
    ari = np.array(ari)

    best_acc_num = acc[idx[:num]].tolist()
    best_nmi_num = nmi[idx[:num]].tolist()
    best_ari_num = ari[idx[:num]].tolist()

    print('acc: %s' % (str(best_acc_num)))
    print('nmi: %s' % (str(best_nmi_num)))
    print('ari: %s' % (str(best_ari_num)))
    return best_acc_num, best_nmi_num, best_ari_num




def train_got(model, L1_reg, L2_reg, optimizer, config, epochs, device):
    # Initialization
    torch.manual_seed(config['training']['got']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training']['got']['seed'])

    L1_tensor = to_tensor(L1_reg, device)
    L2_tensor = to_tensor(L2_reg, device)
    params = wasserstein_initialisation(L1_reg, L2_reg)
    history = []
    for epoch in range(epochs):
        cost = 0
        for iter in range(config['training']['got']['num_iter']):
            eps = torch.randn((model._nodes, model._nodes)).to(device)
            DS = model(eps)
            loss = model.loss_got(L1_tensor, L2_tensor, DS, params)
            cost += loss
        cost = cost / config['training']['got']['num_iter']
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        history.append(cost.item())
        if epoch % 50 == 0:
            print('[Epoch %4d/%d] loss: %f - std: %f' % (epoch, epochs, cost.item(), model.std.detach().mean()))
    P = model.doubly_stochastic(model.mean)
    max_val, max_idx = torch.max(P, dim=1, keepdim=True)
    P_pred = torch.zeros_like(P)
    P_pred.scatter_(1, max_idx, 1)
    return P.detach(), P_pred.detach()


def get_cat_feature(model, x1_train, x2_train):
    with torch.no_grad():
        model.eval()
        x1_recon, x2_recon, feature1, feature2, fake_fea2, fake_fea1 = model(x1_train, x2_train)
        model.train()
    fea_cat1 = torch.cat((feature1, fake_fea2), dim=1)
    fea_cat2 = torch.cat((fake_fea1, feature2), dim=1)
    return fea_cat1, fea_cat2

def get_got_input(fea1, fea2, config, k=100, graph=True):
    # 1. g1 and g2
    g1, g2 = calculate_graphs(fea1, fea2)
    # 2. L1 and L2
    L1 = calculate_laplacian(g1, k=k).cpu().numpy()
    L2 = calculate_laplacian(g2, k=k).cpu().numpy()
    if graph:
        [L1_reg, L2_reg] = regularise_and_invert(L1, L2, config['training']['got']['alpha'], ones=True)
    else:
        L1_reg = L1
        L2_reg = L2
    return g1, g2, L1_reg, L2_reg

def get_global_p(flag, P_part, device='cuda'):
    num_sample = flag.shape[0]
    P = torch.zeros(num_sample, dtype=torch.float32).to(device)
    vec = torch.zeros((num_sample, )).to(device)
    vec[flag] = 1
    P = P + torch.diag(vec)
    idx = []
    for i in range(len(flag)):
        if flag[i] == False:
            idx.append(i)
    for i in range(P_part.shape[0]):
        P[idx[i], idx] = P_part[i]
    return P