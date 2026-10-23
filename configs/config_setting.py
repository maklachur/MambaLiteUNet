from torchvision import transforms
from utils import *

from datetime import datetime
class setting_config:
    """
    config file for training setting.
    """
    network = 'MambaLiteUNet' 
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [16,32,48,64,96,128] 
    }

    # Set data path ('ISIC2017', 'ISIC2018', 'HAM10000', 'PH2', or 'custom')
    datasets = 'ISIC2018'
    if datasets in ('ISIC2017', 'ISIC2018', 'HAM10000', 'PH2', 'custom'):
        data_path = f'./data/{datasets}/'
    else:
        raise Exception('dataset path not correct!! please double check...')

    print(f'Dataset: {datasets} Selected!!')
    print(f"Channel configuration: {model_config['c_list']} Using!!")

    #Loss set up
    criterion = BceDiceLoss()

    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    batch_size = 8
    epochs = 300

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5


    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01
        rho = 0.9
        eps = 1e-6
        weight_decay = 0.05
    elif opt == 'Adagrad':
        lr = 0.01
        lr_decay = 0
        eps = 1e-10
        weight_decay = 0.05
    elif opt == 'Adam':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0001
        amsgrad = False
    elif opt == 'AdamW':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    elif opt == 'Adamax':
        lr = 2e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
    elif opt == 'ASGD':
        lr = 0.01
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
    elif opt == 'RMSprop':
        lr = 1e-2
        momentum = 0
        alpha = 0.99
        eps = 1e-8
        centered = False
        weight_decay = 0
    elif opt == 'Rprop':
        lr = 1e-2
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
        dampening = 0
        nesterov = False
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]
        gamma = 0.1
        last_epoch = -1
    elif sch == 'ExponentialLR':
        gamma = 0.99
        last_epoch = -1
    elif sch == 'CosineAnnealingLR':
        T_max = 50
        eta_min = 0.00001
        last_epoch = -1
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'
        factor = 0.1
        patience = 10
        threshold = 0.0001
        threshold_mode = 'rel'
        cooldown = 0
        min_lr = 0
        eps = 1e-08
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50
        T_mult = 2
        eta_min = 1e-6
        last_epoch = -1
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20