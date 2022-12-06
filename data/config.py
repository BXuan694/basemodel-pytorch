from models import *
from torchvision.datasets import *
import torchvision.models as models

# These are in RGB and are for ImageNet
MEANS = (123.675, 116.28, 123.675)
STD = (58.395, 57.12, 58.395)

Caltech256_CLASSES = ['001.ak47', '002.american-flag', '003.backpack', '004.baseball-bat', '005.baseball-glove', '006.basketball-hoop', '007.bat', '008.bathtub', '009.bear', '010.beer-mug', '011.billiards', '012.binoculars', '013.birdbath', '014.blimp', '015.bonsai-101', '016.boom-box', '017.bowling-ball', '018.bowling-pin', '019.boxing-glove', '020.brain-101', '021.breadmaker', '022.buddha-101', '023.bulldozer', '024.butterfly', '025.cactus', '026.cake', '027.calculator', '028.camel', '029.cannon', '030.canoe', '031.car-tire', '032.cartman', '033.cd', '034.centipede', '035.cereal-box', '036.chandelier-101', '037.chess-board', '038.chimp', '039.chopsticks', '040.cockroach', 
                    '041.coffee-mug', '042.coffin', '043.coin', '044.comet', '045.computer-keyboard', '046.computer-monitor', '047.computer-mouse', '048.conch', '049.cormorant', '050.covered-wagon', '051.cowboy-hat', '052.crab-101', '053.desk-globe', '054.diamond-ring', '055.dice', '056.dog', '057.dolphin-101', '058.doorknob', '059.drinking-straw', '060.duck', '061.dumb-bell', '062.eiffel-tower', '063.electric-guitar-101', '064.elephant-101', '065.elk', '066.ewer-101', '067.eyeglasses', '068.fern', '069.fighter-jet', '070.fire-extinguisher', '071.fire-hydrant', '072.fire-truck', '073.fireworks', '074.flashlight', '075.floppy-disk', '076.football-helmet', '077.french-horn', '078.fried-egg', '079.frisbee', '080.frog', 
                    '081.frying-pan', '082.galaxy', '083.gas-pump', '084.giraffe', '085.goat', '086.golden-gate-bridge', '087.goldfish', '088.golf-ball', '089.goose', '090.gorilla', '091.grand-piano-101', '092.grapes', '093.grasshopper', '094.guitar-pick', '095.hamburger', '096.hammock', '097.harmonica', '098.harp', '099.harpsichord', '100.hawksbill-101', '101.head-phones', '102.helicopter-101', '103.hibiscus', '104.homer-simpson', '105.horse', '106.horseshoe-crab', '107.hot-air-balloon', '108.hot-dog', '109.hot-tub', '110.hourglass', '111.house-fly', '112.human-skeleton', '113.hummingbird', '114.ibis-101', '115.ice-cream-cone', '116.iguana', '117.ipod', '118.iris', '119.jesus-christ', '120.joy-stick',
                    '121.kangaroo-101', '122.kayak', '123.ketch-101', '124.killer-whale', '125.knife', '126.ladder', '127.laptop-101', '128.lathe', '129.leopards-101', '130.license-plate', '131.lightbulb', '132.light-house', '133.lightning', '134.llama-101', '135.mailbox', '136.mandolin', '137.mars', '138.mattress', '139.megaphone', '140.menorah-101', '141.microscope', '142.microwave', '143.minaret', '144.minotaur', '145.motorbikes-101', '146.mountain-bike', '147.mushroom', '148.mussels', '149.necktie', '150.octopus', '151.ostrich', '152.owl', '153.palm-pilot', '154.palm-tree', '155.paperclip', '156.paper-shredder', '157.pci-card', '158.penguin', '159.people', '160.pez-dispenser',
                    '161.photocopier', '162.picnic-table', '163.playing-card', '164.porcupine', '165.pram', '166.praying-mantis', '167.pyramid', '168.raccoon', '169.radio-telescope', '170.rainbow', '171.refrigerator', '172.revolver-101', '173.rifle', '174.rotary-phone', '175.roulette-wheel', '176.saddle', '177.saturn', '178.school-bus', '179.scorpion-101', '180.screwdriver', '181.segway', '182.self-propelled-lawn-mower', '183.sextant', '184.sheet-music', '185.skateboard', '186.skunk', '187.skyscraper', '188.smokestack', '189.snail', '190.snake', '191.sneaker', '192.snowmobile', '193.soccer-ball', '194.socks', '195.soda-can', '196.spaghetti', '197.speed-boat', '198.spider', '199.spoon', '200.stained-glass',
                    '201.starfish-101', '202.steering-wheel', '203.stirrups', '204.sunflower-101', '205.superman', '206.sushi', '207.swan', '208.swiss-army-knife', '209.sword', '210.syringe', '211.tambourine', '212.teapot', '213.teddy-bear', '214.teepee', '215.telephone-box', '216.tennis-ball', '217.tennis-court', '218.tennis-racket', '219.theodolite', '220.toaster', '221.tomato', '222.tombstone', '223.top-hat', '224.touring-bike', '225.tower-pisa', '226.traffic-light', '227.treadmill', '228.triceratops', '229.tricycle', '230.trilobite-101', '231.tripod', '232.t-shirt', '233.tuning-fork', '234.tweezer', '235.umbrella-101', '236.unicorn', '237.vcr', '238.video-projector', '239.washing-machine', '240.watch-101',
                    '241.waterfall', '242.watermelon', '243.welding-mask', '244.wheelbarrow', '245.windmill', '246.wine-bottle', '247.xylophone', '248.yarmulke', '249.yo-yo', '250.zebra', '251.airplanes-101', '252.car-side-101', '253.faces-easy-101', '254.greyhound', '255.tennis-shoes', '256.toad', '257.clutter']
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class Config(object):
    """
    After implement this class, you can call 'cfg.x' instead of 'cfg['x']' to get a certain parameter.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making the changes given by new_config_dict.
        """
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object. Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def __repr__(self):
        return self.name
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

# -------------------------- DATAS -------------------------- #
dataset_base = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'trainImgPrefix': './data/coco/images/',
    'trainRoot':   'path_to_annotation_file',

    # Validation images and annotations.
    'validImgPrefix': './data/coco/images/',
    'validRoot':   'path_to_annotation_file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # Organization of images, "default" represents each folder of images belongs to a class.
    'structure': "default",

    # A list of names for each of you classes.
    'labelMap': [],

    # Num of classes
    'numClses': -1
})

Caltech256_dataset = dataset_base.copy({
   'name': 'Caltech-256',

    'trainRoot': '/wbx/code/basemodel',
    'trainImgPrefix': '',

    'validRoot': '/wbx/code/basemodel',
    'validImgPrefix': '',

    'structure': "txt",
    'labelMap': Caltech256_CLASSES,
    'numClses': len(Caltech256_CLASSES)
})

CIFAR10_dataset = dataset_base.copy({
    'name': 'CIFAR10',

    'trainRoot': '/home/w/data/BL/bl121/labelCOCO/anno.json',
    'trainImgPrefix': '',

    'validRoot': '/home/w/data/BL/bl121/labelCOCO/anno.json',
    'validImgPrefix': '',

    'structure': 'default',
    'label_map': CIFAR10_CLASSES,
    'numClses': len(CIFAR10_CLASSES)

})

# ----------------------- CONFIG DEFAULTS ----------------------- #
base_config = Config({
    'dataset': Caltech256_dataset,
    'num_classes': -1, # This should include the background class

    'train_pipeline':  [
        dict(type='LoadImageFromFile'),                                #read img process 
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  #load annotations 
        dict(type='Resize',                                             #多尺度训练，随即从后面的size选择一个尺寸
            img_scale=[(768, 768)],
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),                    # 随机反转,0.5的概率
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),               
        dict(type='Pad', size_divisor=32),                                #pad另一边的size为32的倍数，solov2对网络输入的尺寸有要求，图像的size需要为32的倍数
        dict(type='DefaultFormatBundle'),                                #将数据转换为tensor，为后续网络计算
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')),   
    ],
    'test_pipeline': [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(768, 768),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ],


})

res18_Caltech256_config = base_config.copy({
    'name': 'resnet_Caltech256',
    'backbone': ResNet18(),
    'model_official': models.resnet18(pretrained=True), # 从官方模型抽取预训练权重
    'dataset': Caltech256_dataset,
    'num_classes': Caltech256_dataset.numClses,
    'inputShape': (384, 384),

    'batchSize': 16,
    'workers_per_gpu': 4,
    # learning policy
    'lr_config': dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.005, step=[14, 19, 23]),
    # optimizer
    'optimizer': dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),  
    #'optimizer_config': dict(grad_clip=dict(max_norm=35, norm_type=2)),   #梯度平衡策略

    'resume_from': "weights/resnet_Caltech256_epoch_1.pth",    #从保存的权重文件中读取，如果为None则权重自己初始化
    'total_epoch': 25,
    'epoch_iters_start': 1,    #本次训练的开始迭代起始轮数
})

res34_Caltech256_config = base_config.copy({
    'name': 'resnet_Caltech256',
    'backbone': ResNet34(),
    'model_official': models.resnet34(pretrained=True), # 从官方模型抽取预训练权重
    'dataset': Caltech256_dataset,
    'num_classes': Caltech256_dataset.numClses,
    'inputShape': (384, 384),

    'batchSize': 16,
    'workers_per_gpu': 4,
    # learning policy
    'lr_config': dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.005, step=[14, 19, 23]),
    # optimizer
    'optimizer': dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),  
    #'optimizer_config': dict(grad_clip=dict(max_norm=35, norm_type=2)),   #梯度平衡策略

    'resume_from': "weights_n1/resnet_Caltech256_epoch_24.pth",    #从保存的权重文件中读取，如果为None则权重自己初始化
    'total_epoch': 25,
    'epoch_iters_start': 1,    #本次训练的开始迭代起始轮数
})

res50_Caltech256_config = base_config.copy({
    'name': 'resnet_Caltech256',
    'backbone': ResNet50(),
    'model_official': models.resnet50(pretrained=True), # 从官方模型抽取预训练权重
    'dataset': Caltech256_dataset,
    'num_classes': Caltech256_dataset.numClses,
    'inputShape': (384, 384),

    'batchSize': 16,
    'workers_per_gpu': 4,
    # learning policy
    'lr_config': dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.005, step=[14, 19, 23]),
    # optimizer
    'optimizer': dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),  
    #'optimizer_config': dict(grad_clip=dict(max_norm=35, norm_type=2)),   #梯度平衡策略

    'resume_from': "weights/resnet_Caltech256_epoch_1.pth",    #从保存的权重文件中读取，如果为None则权重自己初始化
    'total_epoch': 25,
    'epoch_iters_start': 1,    #本次训练的开始迭代起始轮数
})

res101_Caltech256_config = base_config.copy({
    'name': 'resnet_Caltech256',
    'backbone': ResNet101(),
    'model_official': models.resnet101(pretrained=True),
    'dataset': Caltech256_dataset,
    'num_classes': Caltech256_dataset.numClses,
    'inputShape': (384, 384),

    'batchSize': 4,
    'workers_per_gpu': 4,
    # learning policy
    'lr_config': dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.005, step=[14, 19, 23]),
    # optimizer
    'optimizer': dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),  
    #'optimizer_config': dict(grad_clip=dict(max_norm=35, norm_type=2)),   #梯度平衡策略

    'resume_from': "weights/resnet_Caltech256_epoch_1.pth",    #从保存的权重文件中读取，如果为None则权重自己初始化
    'total_epoch': 25,
    'epoch_iters_start': 1,    #本次训练的开始迭代起始轮数
})


cfg = res50_Caltech256_config.copy()


def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]

def set_dataset(dataset_name:str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)
