_base_ = ['coco_transformer.py']

lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
batch_size = 2
weight_decay = 0.0001
epochs = 60
lr_drop = 50
save_checkpoint_interval = 10
clip_max_norm = 0.1

modelname = 'grouppose'
frozen_weights = None
use_checkpoint = False
dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None

# for transformer
hidden_dim = 256
dropout = 0.0
dim_feedforward = 2048
enc_layers = 6
dec_layers = 6
pre_norm = False
return_intermediate_dec = True
enc_n_points = 4
dec_n_points = 4
learnable_tgt_init = False
transformer_activation = 'relu'

# for main model
num_classes=2
nheads = 8
num_queries = 100
num_feature_levels = 4
dec_pred_class_embed_share = False
dec_pred_pose_embed_share = False
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
cls_no_bias = False
num_body_points = 17

# for loss
focal_alpha = 0.25
cls_loss_coef = 2.0
keypoints_loss_coef = 10.0
oks_loss_coef=4.0
interm_loss_coef = 1.0
no_interm_loss = False
aux_loss = True

# for matcher
matcher_type = 'HungarianMatcher'
set_cost_class = 2.0
set_cost_keypoints = 10.0
set_cost_oks=4.0

# for postprocess
num_select = 50

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0
