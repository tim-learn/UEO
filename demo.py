import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
import sklearn.metrics as skm

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

class CoOp_PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_cls = len(classnames)
        ctx_init = 'a photo of a'
        self.n_ctx = len(ctx_init.split(" "))

        if ctx_init:
            # use given words to initialize context vectors
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + self.n_ctx, :].cuda()
            self.prompt_prefix = ctx_init       
        else:
            # random initialization
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        self.get_prefix_suffix_token(classnames, clip_model)

        print('Initial context: {:}, Number of context words (tokens): {:}'.format(self.prompt_prefix, self.n_ctx))
        
    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        _tokenizer = _Tokenizer()
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.type(self.dtype)

        return x

def image_clip_train(resize_size=224):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
    return  transforms.Compose([
        transforms.RandomResizedCrop(size=resize_size, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        normalize
    ])

def image_clip_test(resize_size=224):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
    return  transforms.Compose([
        transforms.Resize(size=resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resize_size),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        normalize
    ])

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def modify_list(args):
    new_tar = []
    txt_tar = open(args.train_dset_path).readlines()
    for i in range(len(txt_tar)):
        rec = txt_tar[i]
        reci = rec.strip().split(' ')
        if int(reci[1]) in args.tar_classes:
            new_tar.append(rec)
    txt_tar1 = new_tar.copy()

    new_tar = []
    txt_tar = open(args.test_dset_path).readlines()
    for i in range(len(txt_tar)):
        rec = txt_tar[i]
        reci = rec.strip().split(' ')
        if int(reci[1]) in args.tst_classes:
            new_tar.append(rec)
    txt_tar2 = new_tar.copy()
    return txt_tar1, txt_tar2

def prepare_dataset(args):
    ## prepare data split
    if args.dset == 'OFFICE':
        domains = ['amazon', 'dslr', 'webcam']
        domain = domains[args.tid] + '/images'
        args.src_classes = [i for i in range(25)]
        args.tst_classes = [i for i in range(31)]

        if args.da == 'CSDA':
            args.src_classes = [i for i in range(31)]
            args.tar_classes = [i for i in range(31)]
            args.tst_classes = [i for i in range(31)]
        elif args.da == 'CDA':
            args.tar_classes = [i for i in range(25)]
        elif args.da == 'PDA':
            args.tar_classes = [i for i in range(20)]     
        elif args.da == 'ODA':
            args.tar_classes = [i for i in range(28)]
        elif args.da == 'OPDA':
            args.tar_classes = [i for i in range(15)] + [i for i in range(25, 28)]

        allclassnames = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 
        'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 
        'phone', 'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 
        'trash_can']

    elif args.dset == 'OFFICEHOME':
        domains = ['Art', 'Clipart', 'Product', 'RealWorld']
        domain = domains[args.tid]

        args.src_classes = [i for i in range(50)]
        args.tst_classes = [i for i in range(65)]

        if args.da == 'CSDA':
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(65)]
            args.tst_classes = [i for i in range(65)]
        elif args.da == 'CDA':
            args.tar_classes = [i for i in range(50)]
        elif args.da == 'PDA':
            args.tar_classes = [i for i in range(35)]
        elif args.da == 'ODA':
            args.tar_classes = [i for i in range(60)]
        elif args.da == 'OPDA':
            args.tar_classes = [i for i in range(35)] + [i for i in range(50, 60)]

        allclassnames = ['alarm_clock', 'backpack', 'batteries', 'bed', 'bike', 'bottle', 'bucket', 'calculator', 'calendar', 'candles', 
        'chair', 'clipboards', 'computer', 'couch', 'curtains', 'desk_lamp', 'drill', 'eraser', 'exit_sign', 'fan',
        'file_cabinet', 'flipflops', 'flowers', 'folder', 'fork', 'glasses', 'hammer', 'helmet', 'kettle', 'keyboard',
        'knives', 'lamp_shade', 'laptop', 'marker', 'monitor', 'mop', 'mouse', 'mug', 'notebook', 'oven', 
        'pan', 'paper_clip', 'pen', 'pencil', 'postit_notes', 'printer', 'push_pin', 'radio', 'refrigerator', 'ruler',
        'scissors', 'screwdriver', 'shelf', 'sink', 'sneakers', 'soda', 'speaker', 'spoon', 'table', 'telephone',
        'toothbrush', 'toys', 'trash_can', 'tv', 'webcam']

    elif args.dset == 'VISDAC':
        domains = ['train', 'validation']
        domain = domains[args.tid]

        args.src_classes = [i for i in range(8)]
        args.tst_classes = [i for i in range(12)]

        if args.da == 'CSDA':
            args.src_classes = [i for i in range(12)]
            args.tar_classes = [i for i in range(12)]
            args.tst_classes = [i for i in range(12)]
        elif args.da == 'CDA':
            args.tar_classes = [i for i in range(8)]
        elif args.da == 'PDA':
            args.tar_classes = [i for i in range(6)]    
        elif args.da == 'ODA':
            args.tar_classes = [i for i in range(10)]
        elif args.da == 'OPDA':
            args.tar_classes = [i for i in range(6)] + [i for i in range(8, 10)]

        allclassnames = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant', 'skateboard',
        'train', 'truck']

    elif args.dset == 'DOMAINNET':
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        domain = domains[args.tid]

        args.src_classes = [i for i in range(300)]
        args.tst_classes = [i for i in range(345)]

        if args.da == 'CSDA':
            args.src_classes = [i for i in range(345)]
            args.tar_classes = [i for i in range(345)]
            args.tst_classes = [i for i in range(345)]
        elif args.da == 'CDA':
            args.tar_classes = [i for i in range(300)]
        elif args.da == 'PDA':
            args.tar_classes = [i for i in range(250)]
        elif args.da == 'ODA':
            args.tar_classes = [i for i in range(330)]
        elif args.da == 'OPDA':
            args.tar_classes = [i for i in range(250)] + [i for i in range(300, 330)]

        allclassnames = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm',
        'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball',
        'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle',
        'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet',
        'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly',
        'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon',
        'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier',
        'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler',
        'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher',
        'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck',
        'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan',
        'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops',
        'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe',
        'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat',
        'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon',
        'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo',
        'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb',
        'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map',
        'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike',
        'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean',
        'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can', 'palm_tree', 'panda', 'pants',
        'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano',
        'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool',
        'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow',
        'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw',
        'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe',
        'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake',
        'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square',
        'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove',
        'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword',
        'syringe', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China',
        'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor',
        'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt', 'umbrella', 'underwear',
        'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle',
        'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

    args.allclassnames = allclassnames  
    if args.dset == 'DOMAINNET':
        args.train_dset_path = os.path.join(args.list_root, args.dset, domains[args.tid] + '_train.txt')
        args.test_dset_path = os.path.join(args.list_root, args.dset, domains[args.tid] + '_test.txt')
    else:
        args.train_dset_path = os.path.join(args.list_root, args.dset, domains[args.tid] + '_list.txt')
        args.test_dset_path = os.path.join(args.list_root, args.dset, domains[args.tid] + '_list.txt')        

    txt_tar, txt_test = modify_list(args)

    ## prepare dataloader
    dsets = {}
    dset_loaders = {}
    dsets['target'] = ImageList_idx(txt_tar, root=os.path.join(args.data_root, args.dset), transform=image_clip_train())
    dset_loaders['target'] = DataLoader(dsets['target'], batch_size=args.bs, shuffle=True, num_workers=args.worker, drop_last=False) # pin_memory=True
    dsets["test"] = ImageList(txt_test, root=os.path.join(args.data_root, args.dset), transform=image_clip_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=args.bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    dsets["tar_test"] = ImageList(txt_tar, root=os.path.join(args.data_root, args.dset), transform=image_clip_test())
    dset_loaders["tar_test"] = DataLoader(dsets["tar_test"], batch_size=args.bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return args, dset_loaders

def get_score(logits, labels, id_label):
    outputs = torch.nn.Softmax(dim=1)(logits)
    scores, p_labels = torch.max(outputs, dim=1)

    matrix = skm.confusion_matrix(labels[labels < id_label].numpy(), p_labels[labels < id_label].numpy())
    global_acc = matrix.diagonal().sum()/ matrix.sum()
    class_acc = matrix.diagonal() / (matrix.sum(axis=1) + 1e-10)
    id_score = class_acc[matrix.sum(axis=1) > 0].mean()

    if sum(labels >= id_label) > 0:
        ood_labels = (labels < id_label).float()
        fpr, tpr, thresholds = skm.roc_curve(ood_labels, scores.numpy())
        auc_score = skm.auc(fpr, tpr)
        return [global_acc, id_score, auc_score]
    else:
        return [id_score, global_acc]

def norm_feature(features):
    features = features/ features.norm(dim=-1, keepdim=True)
    return features

def load_features(vmodel, loader, text_features):
    logits_, labels_ = [], []
    with torch.no_grad():
        for _, data in enumerate(loader):
            images = data[0].cuda()
            labels = data[1]
            image_features = vmodel(images)
            image_features = norm_feature(image_features)
            # clip_logits = 100. * image_features @ text_features.T
            clip_logits = image_features @ text_features.T
            logits_.append(clip_logits)
            labels_.append(labels)
    logits_, labels_ = torch.cat(logits_), torch.cat(labels_)   
    
    return logits_, labels_

def loss_entropy(input_, average=True):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if entropy.dim() == 1:
        entropy = torch.sum(entropy)
        return entropy

    if average:
        entropy = torch.sum(entropy, dim=1).mean()
    else:
        entropy = torch.sum(entropy, dim=1)
    return entropy 

def loss_entropy_wei(input_, weight):
    epsilon = 1e-5
    entropy = torch.sum(-input_ * torch.log(input_ + epsilon), dim=1)
    entropy = torch.sum(entropy * weight) / torch.sum(weight)
    return entropy 

def compute_transport_loss(logits, sim_t):
    s_dist = torch.nn.Softmax(dim=1)(logits)
    t_dist = torch.nn.Softmax(dim=0)(logits)
    cost = 1 - sim_t
    s_cost = (cost * s_dist).sum(1).mean()
    t_cost = (cost * t_dist).sum(0).mean()
    return s_cost + t_cost

class VisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, x):
        x = self.model(x.type(self.dtype))
        return x

def train_clip(args, dset_loaders):
    # load network
    clip_model, _ = clip.load(args.net)
    text_encoder = TextEncoder(clip_model).cuda()
    visual_encoder = VisualEncoder(clip_model).cuda()
    known_classnames = [args.allclassnames[i] for i in range(len(args.src_classes))]
    coop_prompt_learner = CoOp_PromptLearner(known_classnames, clip_model)

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    visual_encoder.eval()
    for p in visual_encoder.parameters():
        p.requires_grad = False

    coop_prompt_learner.eval()
    for p in coop_prompt_learner.parameters():
        p.requires_grad = False

    # --------------------------------------
    label_maps = torch.zeros(1 + max(max(args.src_classes), max(args.tst_classes)), )
    for i in range(len(args.src_classes)):
        label_maps[args.src_classes[i]] = i
    k = 0

    for i in range(len(args.tst_classes)):
        if not args.tst_classes[i] in args.src_classes:
            label_maps[args.tst_classes[i]] = len(args.src_classes) + k
            k += 1

    args.label_maps = label_maps

    print('src private classes: {:}'.format(set(args.src_classes) - set(args.tar_classes)))
    print('tar private classes: {:}'.format(set(args.tar_classes) - set(args.src_classes)))
    print('shared classes: {:}'.format(set(args.tar_classes) & set(args.src_classes)))

    log_str = ('Training Epoch: {:} / {:}'.format(0, args.epochs))
    with torch.no_grad():
        prompts = coop_prompt_learner()
        tokenized_prompts = coop_prompt_learner.tokenized_prompts
        text_features = text_encoder(prompts, tokenized_prompts)
        text_features = norm_feature(text_features)

        clip_logits, images_labels = load_features(visual_encoder, dset_loaders['test'], text_features)
        images_labels = label_maps[images_labels].long()

        _score = get_score(clip_logits.cpu().float(), images_labels, len(args.src_classes))

    if len(set(args.tst_classes) - set(args.src_classes)):
        log_str += ('\n(Tar-test)  GACC:{:.2f} PACC:{:.2f} AUROC:{:.2f}'.format(_score[0]*100, _score[1]*100, _score[2]*100))
    else:
        log_str += ('\n(Tar-test)  PACC:{:.2f} GACC:{:.2f}'.format(_score[0]*100, _score[1]*100))

    with torch.no_grad():
        prompts = coop_prompt_learner()
        tokenized_prompts = coop_prompt_learner.tokenized_prompts
        text_features = text_encoder(prompts, tokenized_prompts)
        text_features = norm_feature(text_features)

        clip_logits, images_labels = load_features(visual_encoder, dset_loaders['tar_test'], text_features)
        images_labels = label_maps[images_labels].long()

        _score = get_score(clip_logits.cpu().float(), images_labels, len(args.src_classes))

    if len(set(args.tar_classes) - set(args.src_classes)):
        log_str += ('\n(Tar-train) GACC:{:.2f} PACC:{:.2f} AUROC:{:.2f}'.format(_score[0]*100, _score[1]*100, _score[2]*100))
    else:
        log_str += ('\n(Tar-train) PACC:{:.2f} GACC:{:.2f}'.format(_score[0]*100, _score[1]*100))

    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    param_group = []
    if args.plr > 0:
        for k, v in coop_prompt_learner.named_parameters():
            v.requires_grad = True
            param_group += [{'params': v, 'lr': args.plr}]
    if args.vlr > 0:
        for k, v in visual_encoder.named_parameters():
            if 'bn' in k or 'ln' in k:
                v.requires_grad = True
                param_group += [{'params': v, 'lr': args.vlr}]

    optimizer = torch.optim.SGD(param_group)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(dset_loaders['target']))

    target_outputs = torch.nn.Softmax(dim=1)(clip_logits)
    target_maxpro, _ = torch.max(target_outputs, dim=1)

    for epoch_idx in range(args.epochs):
        visual_encoder.eval()

        for i, data in enumerate(dset_loaders['target']):
            images = data[0].cuda()
            index_t = data[2].cuda()
            
            image_features = visual_encoder(images)
            image_features = norm_feature(image_features)

            prompts = coop_prompt_learner()
            tokenized_prompts = coop_prompt_learner.tokenized_prompts
            text_features = text_encoder(prompts, tokenized_prompts)
            text_features = norm_feature(text_features)
            logits = 100. * image_features @ text_features.T

            outputs = torch.nn.Softmax(dim=1)(logits)
            if args.noweight:
                mean_outputs = torch.mean(outputs, dim=0)
                loss = loss_entropy(outputs) - args.trade * loss_entropy(mean_outputs)
            else:
                if args.oracle:
                    images_labels = label_maps[data[1]].long().cuda()
                    weight = (images_labels < len(args.src_classes)).type(target_maxpro.dtype) + 1e-3
                    mean_outputs = torch.mm(torch.diag(1 / weight), outputs).sum(dim=0) / torch.sum(1 / weight)
                    loss = loss_entropy_wei(outputs, weight) - args.trade * loss_entropy(mean_outputs)
                else:
                    weight = target_maxpro[index_t]
                    mean_outputs = torch.mm(torch.diag(1 / weight), outputs).sum(dim=0) / torch.sum(1 / weight)
                    loss = loss_entropy_wei(outputs, weight) - args.trade * loss_entropy(mean_outputs)

            optimizer.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if (epoch_idx + 1) % args.eval_epoch == 0:
            log_str = ('Training Epoch: {:} / {:}'.format(epoch_idx + 1, args.epochs))
            visual_encoder.eval()
            with torch.no_grad():
                prompts = coop_prompt_learner()
                tokenized_prompts = coop_prompt_learner.tokenized_prompts
                text_features = text_encoder(prompts, tokenized_prompts)
                text_features = norm_feature(text_features)

                clip_logits, images_labels = load_features(visual_encoder, dset_loaders['test'], text_features)
                images_labels = label_maps[images_labels].long()

                _score = get_score(clip_logits.cpu().float(), images_labels, len(args.src_classes))

            if len(set(args.tst_classes) - set(args.src_classes)):
                log_str += ('\n(Tar-test)  GACC:{:.2f} PACC:{:.2f} AUROC:{:.2f}'.format(_score[0]*100, _score[1]*100, _score[2]*100))
            else:
                log_str += ('\n(Tar-test)  PACC:{:.2f} GACC:{:.2f}'.format(_score[0]*100, _score[1]*100))

            with torch.no_grad():
                prompts = coop_prompt_learner()
                tokenized_prompts = coop_prompt_learner.tokenized_prompts
                text_features = text_encoder(prompts, tokenized_prompts)
                text_features = norm_feature(text_features)

                clip_logits, images_labels = load_features(visual_encoder, dset_loaders['tar_test'], text_features)
                images_labels = label_maps[images_labels].long()

                _score = get_score(clip_logits.cpu().float(), images_labels, len(args.src_classes))

            if len(set(args.tar_classes) - set(args.src_classes)):
                log_str += ('\n(Tar-train) GACC:{:.2f} PACC:{:.2f} AUROC:{:.2f}'.format(_score[0]*100, _score[1]*100, _score[2]*100))
            else:
                log_str += ('\n(Tar-train) PACC:{:.2f} GACC:{:.2f}'.format(_score[0]*100, _score[1]*100))

            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]));
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T
        self.memory =  self.memory.cuda()

    def forward(self, x, y):
        # pdb.set_trace()
        out = torch.mm(x.float(), self.memory.t())/self.T
        return out

    def update_weight(self, features, index):
        if not self.flag:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(0.0)
            weight_pos.add_(torch.mul(features.data, 1.0))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
            self.flag = 1
        else:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
        self.memory = torch.nn.functional.normalize(self.memory)#.cuda()

    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xxx')
    parser.add_argument('--dset', type=str, default='OFFICEHOME', choices=['VISDAC', 'OFFICE', 'OFFICEHOME', 'miniDOMAINNET', 'DOMAINNET'])
    parser.add_argument('--tid', type=int, default=0, help="target")
    parser.add_argument('--da', type=str, default='OPDA', choices=['CDA', 'PDA', 'ODA', 'OPDA', 'CSDA'])
    parser.add_argument('--net', type=str, default='RN50', choices=['RN50', 'RN101',  'ViT-B/32', 'ViT-B/16'])
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--data_root', type=str, default='/data1/xxx/datasets/cls/')
    parser.add_argument('--list_root', type=str, default='./list/')

    parser.add_argument('--log', type=str, default='logszz/')
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--plr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--vlr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=50, help="number of epochs")
    parser.add_argument('--eval_epoch', type=int, default=10, help="the interval of evaluation epochs")
    parser.add_argument('--trade', type=float, default=1.0)
    parser.add_argument('--noweight', action="store_true")
    parser.add_argument('--oracle', action="store_true")

    args = parser.parse_args()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
    args, dset_loaders = prepare_dataset(args)
    envs = print_args(args)

    name = args.net.replace("/", "")
    if args.noweight:
        ff = '_noweight'
    elif args.oracle:
        ff = '_oracle'
    else:
        ff = '_ours'

    output_dir_src = osp.join(args.log, str(args.seed) + ff, name + '_vlr_' + str(args.vlr) + '_plr_' + str(args.plr), args.dset)
    if not osp.exists(output_dir_src):
        os.system('mkdir -p ' + output_dir_src)
    if not osp.exists(output_dir_src):
        os.mkdir(output_dir_src)

    args.out_file = open(osp.join(output_dir_src, '@{:}_{:}_trade_{:.1f}.txt'.format(args.tid, args.da, args.trade)), 'w')
    train_clip(args, dset_loaders)

    args.out_file.write('\n' + envs)
    args.out_file.flush()
    args.out_file.close()