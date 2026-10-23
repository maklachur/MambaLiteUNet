import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )



def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()
    
def save_predicted_mask(msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Process the predicted mask
    if datasets == 'retinal':
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    # Plot and save the predicted mask
    plt.figure()
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '_pred.png')
    plt.close()

def save_all_individually(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    """
    Save individual images: input image, ground truth mask, and predicted mask.
    
    Args:
        img (numpy array): The input image.
        msk (numpy array): Ground truth mask.
        msk_pred (numpy array): Predicted mask.
        i (int): Index for saving the file.
        save_path (str): Directory to save the images.
        datasets (str): Type of dataset (e.g., 'retinal').
        threshold (float, optional): Threshold for converting prediction to binary. Defaults to 0.5.
        test_data_name (str, optional): Additional prefix for saved files.
    """
    # Process and normalize image if necessary
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img

    # Process ground truth and predicted masks
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    # Save input image
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    img_filename = f"{save_path}{test_data_name + '_' if test_data_name else ''}{i}_img.png"
    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save ground truth mask
    plt.figure()
    plt.imshow(msk, cmap='gray')
    plt.axis('off')
    gt_filename = f"{save_path}{test_data_name + '_' if test_data_name else ''}{i}_gt.png"
    plt.savefig(gt_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save predicted mask
    plt.figure()
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')
    pred_filename = f"{save_path}{test_data_name + '_' if test_data_name else ''}{i}_pred.png"
    plt.savefig(pred_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    

def _iter_mask2d(arr):
    """Yield 2D masks from one network output or label array.

    Evaluation stores each batch item as (N, H, W) after squeezing the
    class channel, or as (H, W) for a single map. HD95 is per map.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        yield arr
        return
    if arr.ndim == 3:
        for index in range(arr.shape[0]):
            yield arr[index]
        return
    raise ValueError(f'HD95 expects a 2D or 3D mask array, got shape {arr.shape}')


def _binary_surface(mask):
    """Object boundary at 4-connectivity. A one-pixel object is its own surface."""
    from scipy.ndimage import binary_erosion

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    surface = mask & ~binary_erosion(mask, structure=None, border_value=0)
    if not surface.any():
        return mask
    return surface


def _surface_distances(pred_mask, gt_mask):
    """Directed surface distances for a binary pair.

    Returns ``(pred_to_gt, gt_to_pred)``. Both-empty maps are distance 0
    and are reported as empty arrays plus ``both_empty=True``. One-empty
    maps are undefined. Shapes must already match.
    """
    from scipy.ndimage import distance_transform_edt

    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f'Surface-distance shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}'
        )
    if not pred_mask.any() and not gt_mask.any():
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), True
    if not pred_mask.any() or not gt_mask.any():
        return None, None, False

    pred_surface = _binary_surface(pred_mask)
    gt_surface = _binary_surface(gt_mask)
    pred_to_gt = distance_transform_edt(~gt_surface)[pred_surface]
    gt_to_pred = distance_transform_edt(~pred_surface)[gt_surface]
    return pred_to_gt, gt_to_pred, False


def hd95_binary(pred_mask, gt_mask):
    """Symmetric 95th-percentile surface Hausdorff distance, in pixels.

    Both-empty maps are distance 0. One-empty maps are undefined (NaN),
    matching the usual medical-segmentation convention that HD95 needs a
    surface on both sides. This is a SciPy EDT implementation. It is not
    asserted to match medpy bit for bit, and it does not apply millimetre
    spacing because the loader does not provide any.
    """
    pred_to_gt, gt_to_pred, both_empty = _surface_distances(pred_mask, gt_mask)
    if both_empty:
        return 0.0
    if pred_to_gt is None:
        return float('nan')
    distances = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(distances, 95))


def assd_binary(pred_mask, gt_mask):
    """Average symmetric surface distance, in pixels.

    The mean of the concatenated directed surface distances. Both-empty
    maps are 0. One-empty maps are undefined (NaN). Same surface definition
    as ``hd95_binary``. No millimetre spacing.
    """
    pred_to_gt, gt_to_pred, both_empty = _surface_distances(pred_mask, gt_mask)
    if both_empty:
        return 0.0
    if pred_to_gt is None:
        return float('nan')
    distances = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.mean(distances))


def _summarize_surface(preds, gts, distance_fn, threshold=0.5, gt_threshold=0.5, label='surface'):
    """Mean of a per-map surface distance. Missing SciPy is status 'unavailable'."""
    try:
        import scipy.ndimage  # noqa: F401
    except ImportError:
        return {
            'status': 'unavailable',
            'reason': 'scipy is not installed',
            'mean': None,
            'count': 0,
            'undefined': 0,
            'total': 0,
        }

    values = []
    undefined = 0
    total = 0
    for pred, gt in zip(preds, gts):
        pred_maps = list(_iter_mask2d(pred))
        gt_maps = list(_iter_mask2d(gt))
        if len(pred_maps) != len(gt_maps):
            raise ValueError(
                f'{label} batch mismatch: {len(pred_maps)} preds vs {len(gt_maps)} labels'
            )
        for pred_map, gt_map in zip(pred_maps, gt_maps):
            total += 1
            distance = distance_fn(pred_map >= threshold, gt_map >= gt_threshold)
            if np.isnan(distance):
                undefined += 1
            else:
                values.append(distance)

    mean = float(np.mean(values)) if values else float('nan')
    return {
        'status': 'ok',
        'reason': '',
        'mean': mean,
        'count': len(values),
        'undefined': undefined,
        'total': total,
    }


def summarize_hd95(preds, gts, threshold=0.5, gt_threshold=0.5):
    """Mean HD95 over paired prediction / label arrays.

    Returns a dict and never raises ImportError. Missing SciPy yields
    status 'unavailable' so the existing confusion-matrix metrics can
    still be logged. Other failures are the caller's responsibility.
    """
    return _summarize_surface(
        preds, gts, hd95_binary, threshold=threshold, gt_threshold=gt_threshold, label='HD95'
    )


def summarize_assd(preds, gts, threshold=0.5, gt_threshold=0.5):
    """Mean ASSD over paired prediction / label arrays. Same contract as HD95."""
    return _summarize_surface(
        preds, gts, assd_binary, threshold=threshold, gt_threshold=gt_threshold, label='ASSD'
    )


def format_hd95(summary):
    """Short log fragment. Does not include the 'hd95:' key."""
    if not summary or summary.get('status') != 'ok':
        reason = (summary or {}).get('reason', 'not computed')
        return f'n/a ({reason})'
    mean = summary.get('mean')
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        undefined = summary.get('undefined', 0)
        return f'n/a (no defined pairs, undefined={undefined})'
    text = f'{mean:.4f}'
    undefined = summary.get('undefined') or 0
    if undefined:
        text += f' (undefined={undefined})'
    return text


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd
        print(f'BceDiceLoss wb = {self.wb} and wd = {self.wd}')

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

        
