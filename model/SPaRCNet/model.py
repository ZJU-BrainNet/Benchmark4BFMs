from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from torch import optim
from model.model_config import ModelPathArgs

class SPaRCNet_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 128
        return args

    @staticmethod
    def clsf_loss_func(args):
        if args.weights is None:
            ce_weight = [1.0 for _ in range(args.n_class)]
        else:
            ce_weight = args.weights
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.Adam([{'params': model.parameters(), 'lr': args.model_lr}],
                                    betas=(0.9, 0.999), eps=1e-08)

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)
    

class _DenseLayer(nn.Sequential):
	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, conv_bias, batch_norm):
		super(_DenseLayer, self).__init__()
		if batch_norm:
			self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
		# self.add_module('relu1', nn.ReLU()),
		self.add_module('elu1', nn.ELU()),
		self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=conv_bias)),
		if batch_norm:
			self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
		# self.add_module('relu2', nn.ReLU()),
		self.add_module('elu2', nn.ELU()),
		self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=conv_bias)),
		# self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=7, stride=1, padding=3, bias=conv_bias)),
		self.drop_rate = drop_rate

	def forward(self, x):
		# print("Dense Layer Input: ")
		# print(x.size())
		new_features = super(_DenseLayer, self).forward(x)
		# print("Dense Layer Output:")
		# print(new_features.size())
		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
		return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
	def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, conv_bias, batch_norm):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, conv_bias, batch_norm)
			self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
	def __init__(self, num_input_features, num_output_features, conv_bias, batch_norm):
		super(_Transition, self).__init__()
		if batch_norm:
			self.add_module('norm', nn.BatchNorm1d(num_input_features))
		# self.add_module('relu', nn.ReLU())
		self.add_module('elu', nn.ELU())
		self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=conv_bias))
		self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseNetEnconder(nn.Module):
	def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4),  #block_config=(6, 12, 24, 48, 24, 20, 16),  #block_config=(6, 12, 24, 16),
				 in_channels=16, num_init_features=64, bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False):

		super(DenseNetEnconder, self).__init__()

		# First convolution
		first_conv = OrderedDict([('conv0', nn.Conv1d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=conv_bias))])
		# first_conv = OrderedDict([('conv0', nn.Conv1d(in_channels, num_init_features, groups=in_channels, kernel_size=7, stride=2, padding=3, bias=conv_bias))])
		# first_conv = OrderedDict([('conv0', nn.Conv1d(in_channels, num_init_features, kernel_size=15, stride=2, padding=7, bias=conv_bias))])

		# first_conv = OrderedDict([
		# 	('conv0-depth', nn.Conv1d(in_channels, 32, groups=in_channels, kernel_size=7, stride=2, padding=3, bias=conv_bias)),
		# 	('conv0-point', nn.Conv1d(32, num_init_features, kernel_size=1, stride=1, bias=conv_bias)),
		# ])

		if batch_norm:
			first_conv['norm0'] = nn.BatchNorm1d(num_init_features)
		# first_conv['relu0'] = nn.ReLU()
		first_conv['elu0'] = nn.ELU()
		first_conv['pool0'] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

		self.densenet = nn.Sequential(first_conv)

		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
								bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, conv_bias=conv_bias, batch_norm=batch_norm)
			self.densenet.add_module('denseblock%d' % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, conv_bias=conv_bias, batch_norm=batch_norm)
				self.densenet.add_module('transition%d' % (i + 1), trans)
				num_features = num_features // 2

		# Final batch norm
		if batch_norm:
			self.densenet.add_module('norm{}'.format(len(block_config) + 1), nn.BatchNorm1d(num_features))
		# self.features.add_module('norm5', BatchReNorm1d(num_features))

		self.densenet.add_module('relu{}'.format(len(block_config) + 1), nn.ReLU())
		self.densenet.add_module('pool{}'.format(len(block_config) + 1), nn.AvgPool1d(kernel_size=7, stride=3))  # stride originally 1

		self.num_features = num_features

		# Official init from torch repo.
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight.data)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def forward(self, x):

		features = self.densenet(x)
		# print("Final Output")
		# print(features.size())
		return features.view(features.size(0), -1)


class DenseNetClassifier(nn.Module):
	# def __init__(self, growth_rate=16, block_config=(3, 6, 12, 8),  #block_config=(6, 12, 24, 48, 24, 20, 16),  #block_config=(6, 12, 24, 16),
	# 			 in_channels=16, num_init_features=32, bn_size=2, drop_rate=0, conv_bias=False, drop_fc=0.5, num_classes=6):
	def __init__(self, args, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4),
				 num_init_features=64, bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False, drop_fc=0.5):

		super(DenseNetClassifier, self).__init__()

		self.features = DenseNetEnconder(growth_rate=growth_rate, block_config=block_config, in_channels=args.cnn_in_channels,
										 num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate,
										 conv_bias=conv_bias, batch_norm=batch_norm)

		# Linear layer
		self.classifier = nn.Sequential(
			nn.Dropout(p=drop_fc),
			nn.Linear(self.features.num_features*args.cnn_in_channels, args.n_class)
		)

		# Official init from torch repo.
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight.data)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()  
    

	def forward(self, x):
		features = self.features(x)
		out = self.classifier(features)
		return out, features


class SPaRCNet(nn.Module):
    def __init__(self, args: Namespace,):
        super(SPaRCNet, self).__init__()
        self.model = DenseNetClassifier(args=args)
        # checkpoint = torch.load(ModelPathArgs.SPaRCNet_path, map_location=f'cuda:{args.gpu_id}')
        # self.model.load_state_dict(checkpoint, strict=False) 
    
    def forward(self, x):
        return self.model(x)	 
 
    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        logit, features = model(x)
        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y
