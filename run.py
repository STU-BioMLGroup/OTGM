import argparse
import collections
import random
from model import OTGM
from train import pretrain, train
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config
dataset = {
    0: "HandWritten",
    1: "Scene_15",
    2: "BDGP",
    3: "Caltech101-7",
    4: "Caltech101-20",
    5: "Reuters_dim10",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='3', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='1', help='number of test times')

args = parser.parse_args()
dataset = dataset[args.dataset]

def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger = get_logger()

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    # Load data
    X_list, Y_list = load_data(config)
    x1_train = torch.from_numpy(X_list[0]).float().to(device)
    x2_train = torch.from_numpy(X_list[1]).float().to(device)
    config['Autoencoder']['arch1'][0] = x1_train.shape[1]
    config['Autoencoder']['arch2'][0] = x2_train.shape[1]
    flag = get_aligned(x1_train.shape[0], config['training']['aligned_ratio'])
    accumulated_metrics = collections.defaultdict(list)
    config['training']['num_sample'] = x1_train.shape[0]
    config['training']['num_classes'] = len(np.unique(Y_list))

    for data_seed in range(1, args.test_time + 1):
        # Get the Mask
        np.random.seed(data_seed)
        # Set random seeds
        seed = config['training']['seed']
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Build the model
        model = OTGM(config)
        model.to(device)
        optimizer = torch.optim.Adam([{'params': model.network.parameters()},
                                      {'params': model.got.parameters(), 'lr': config['training']['got']['lr']}
                                      ], lr=config['training']['lr'])
        logger.info(model.network.autoencoder1)
        logger.info(model.network.generator1)
        logger.info(optimizer)
        # pretrain
        pretrain_dir = 'pretrain/%s/rate_%s' % (config['dataset'], config['training']['aligned_ratio'])
        if not os.path.exists(pretrain_dir):
            os.mkdir(pretrain_dir)
        pretrain_name = '%s_alpha_%s_l1_%s_l2_%s_%s.pkl' % (config['dataset'], config['training']['alpha'], config['training']['lambda1'],
            config['training']['lambda2'], config['training']['pre_name'])
        pretrain_path = os.path.join(pretrain_dir, pretrain_name)
        if not os.path.exists(pretrain_path):
            pretrain(model.network, optimizer, config, x1_train, x2_train, flag, Y_list, logger, pretrain_path=pretrain_path,
                     device=device)
        else:
            model.network.load_state_dict(torch.load(pretrain_path))
        # shuffle
        model.network.evaluation(logger, x1_train, x2_train, Y_list)
        x1_train, x2_train, P_index, index_mis_aligned, P_gt = get_mis_aligned(x1_train, x2_train, flag, device)
        # train
        acc, nmi, ari = train(model, optimizer, config, x1_train, x2_train, flag, Y_list, index_mis_aligned, P_gt, logger, device)
        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['nmi'].append(nmi)
        accumulated_metrics['ari'].append(ari)

    logger.info('--------------------Training over--------------------')
    cal_std(logger, accumulated_metrics)


if __name__ == '__main__':
    main()
