#just for SwinTransformer, no pretrained
import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

#Base config files
_C.BASE = ['']

#------------------------------------
# Data settings
#------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 64
# Path to dataset
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image(random, bilinear, bicubic)插值
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset,could be overwritten
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU
_C.DATA.PIN_MEMORY = True
# Nuber of data loading threads
_C.DATA.NUM_WORKERS = 1

#------------------------------------
# Model settings
#------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_patch2_window4_32_cifar-10'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight,could be overwritten
_C.MODEL.PRETRAINED = ''
#  Checkpoint to resume, could be overwritten
_C.MODEL.RESUME = ''
# Number of classed, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

#------------------------------------
# Training settings
#------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps, command line
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory, command line
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in CosineLRscheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = None

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon 防止除0错误
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

#------------------------------------
# Augmentation settings
#------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'none'
# Random erase prob
_C.AUG.REPROB = 0.25
# random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# 强度参数
_C.AUG.MIXUP = 0.0
#
_C.AUG.CUTMIX = 0.0
#
_C.AUG.CUTMIX_MINMAX = None
#
_C.AUG.MIX_PROB = 1.0
#
_C.AUG.MIXUP_SWITCH_PROB = 0.5
#
_C.AUG.MIXUP_MODE = 'batch'

#------------------------------------
# Testing settings
#------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = False
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = True
_C.TEST.SHUFFLE = False

#------------------------------------
# misc
#------------------------------------
# Enable Pytorch automatic mixed precision(amp
_C.AMP_ENABLE = False
# [Deprecated]
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = './output'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'train'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 50
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    #检查是否有属性及其值
    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        else:
            return False

    # merge from specific aruguments
    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
        ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

        # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config