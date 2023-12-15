from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.nn import functional as F
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace

class ModelParameters(ABC):
	@abstractmethod
	def getName(self) -> str:
		pass

	@abstractmethod
	def instantiate_new_model(self) -> Module:
		pass

	@abstractmethod
	def get_configuration_space(self) -> ConfigurationSpace:
		pass

	@abstractmethod
	def set_from_configuration(self, config : Configuration):
		pass

@dataclass
class TestNetParameters(ModelParameters):
	pad1: int = 1
	channel1: int = 5
	pad2: int = 1
	channel2: int = 5
	fc2: int = 10

	def getName(self):
		return (self.__class__.__name__ +
				str(self.pad1) + "_" +
				str(self.channel1) + "_" +
				str(self.pad2) + "_" +
				str(self.channel2) + "_" +
				str(self.fc2) + "_")

	def instantiate_new_model(self):
		return TestNet(self)

	def get_configuration_space(self) -> ConfigurationSpace:
		return ConfigurationSpace({
			"pad1": (1, 3),
			"channel1": (1, 20),
			"pad2": (1, 3),
			"channel2": (1, 20),
			"fc2": (3, 20)
		})

	def set_from_configuration(self, config : Configuration):
		self.pad1=config["pad1"]
		self.channel1=config["channel1"]
		self.pad2=config["pad2"]
		self.channel2=config["channel2"]
		self.fc2=config["fc2"]


class TestNet(Module):
	# hyper parameters: pad1 <= 3, channel1 <= 10, pad2 <= 3, channel2 <= 10, fc2: 5 hyperparameters
	# num parameters learned: kern^2 (9, 25, 49) * channel + kern^2 * channel + 7^2*channel * fc2 + fc2
	# keep channel <= 10. start with ~5
	def __init__(self, parameters : TestNetParameters):
		# call the parent constructor
		super(TestNet, self).__init__()

		# initialize first set of CONV => RELU => POOL layers
		kernel_pad_1 = parameters.pad1
		kernel_size_1 = 2*kernel_pad_1 + 1
		out_channels_1 = parameters.channel1
		# 7x5 -> 7x20
		self.conv1 = Conv2d(
						in_channels=5,
						out_channels=out_channels_1,
						padding=kernel_pad_1,
						kernel_size=(kernel_size_1, kernel_size_1))
		self.act1 = ReLU()
		# self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # dim - 2

		# initialize second set of CONV => RELU => POOL layers
		kernel_pad_2 = parameters.pad2
		kernel_size_2 = 2*kernel_pad_2 + 1
		out_channels_2 = parameters.channel2
		# 7x20 -> 7x50
		self.conv2 = Conv2d(
						in_channels=out_channels_1,
						out_channels=out_channels_2,
						padding=kernel_pad_2,
						kernel_size=(kernel_size_2, kernel_size_2))
		self.act2 = ReLU()
		# self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize first (and only) set of FC => RELU layers
		fc1_features = 7 * 7 * out_channels_2
		fc2_features = parameters.fc2
		self.fc1 = Linear(in_features=fc1_features, out_features=fc2_features)
		self.act3 = ReLU()
		# Assume normalization is already done
		self.fc2 = Linear(in_features=fc2_features, out_features=1)

	# x will be 7x7x5, onehot encoded
	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.act1(x)
		# x = self.pool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.act2(x)
		# x = self.pool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		# print(x.shape)
		x = self.act3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		# print(x.shape)
		# output = self.logSoftmax(x)
		# return the output predictions
		x = flatten(x)
		return x

@dataclass
class WideNetParameters(ModelParameters):
	pad1: int = 1
	channel1: int = 5
	width_pad1: int = 2
	pad2: int = 1
	channel2: int = 5
	width_pad2: int = 2
	fc2: int = 10
	fc3: int = 10

	def getName(self):
		return (self.__class__.__name__ +
				str(self.pad1) + "_" +
				str(self.channel1) + "_" +
				str(self.width_pad1) + "_" +
				str(self.pad2) + "_" +
				str(self.channel2) + "_" +
				str(self.width_pad2) + "_" +
				str(self.fc2) + "_" +
				str(self.fc3) + "_")

	def instantiate_new_model(self):
		return WideNet(self)

	def get_configuration_space(self) -> ConfigurationSpace:
		return ConfigurationSpace({
			"pad1": (2, 3),
			"channel1": (15, 40),
			"width_pad1": (3, 5),
			"pad2": (2, 3),
			"channel2": (15, 40),
			"width_pad2": (3, 5),
			"fc2": (20, 40),
			"fc3": (20, 40)
		})

	def set_from_configuration(self, config : Configuration):
		self.pad1=config["pad1"]
		self.channel1=config["channel1"]
		self.width_pad1=config["width_pad1"]
		self.pad2=config["pad2"]
		self.channel2=config["channel2"]
		self.width_pad2=config["width_pad2"]
		self.fc2=config["fc2"]
		self.fc3=config["fc3"]

class WideNet(Module):
	# hyper parameters: pad1 <= 3, channel1 <= 10, pad2 <= 3, channel2 <= 10, fc2: 5 hyperparameters
	# num parameters learned: kern^2 (9, 25, 49) * channel + kern^2 * channel + 7^2*channel * fc2 + fc2
	# keep channel <= 10. start with ~5
	def __init__(self, parameters : WideNetParameters):
		# call the parent constructor
		super(WideNet, self).__init__()

		# initialize first set of CONV => RELU => POOL layers
		layer_width_0 = 7
		kernel_pad_1 = parameters.pad1
		kernel_size_1 = 2*kernel_pad_1 + 1
		min_layer_width_1 = layer_width_0 + 1 - kernel_size_1
		if (min_layer_width_1 < 3):
			min_layer_width_1 = 3
		layer_width_1 = min_layer_width_1 + 2 * 2 * parameters.width_pad1
		layer_pad_1 = int((kernel_size_1 + layer_width_1 - 1 - layer_width_0) / 2)
		out_channels_1 = parameters.channel1
		# 7x5 -> 7x20
		self.conv1 = Conv2d(
						in_channels=5,
						out_channels=out_channels_1,
						padding=layer_pad_1,
						kernel_size=(kernel_size_1, kernel_size_1))
		self.act1 = ReLU()
		# self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # dim - 2

		# initialize second set of CONV => RELU => POOL layers
		kernel_pad_2 = parameters.pad2
		kernel_size_2 = 2*kernel_pad_2 + 1
		min_layer_width_2 = layer_width_1 + 1 - kernel_size_2
		if (min_layer_width_2 < 3):
			min_layer_width_2 = 3
		layer_width_2 = min_layer_width_2 + 2 * parameters.width_pad2
		layer_pad_2 = int((kernel_size_2 + layer_width_2 - 1 - layer_width_1) / 2)
		out_channels_2 = parameters.channel2
		# 7x20 -> 7x50
		self.conv2 = Conv2d(
						in_channels=out_channels_1,
						out_channels=out_channels_2,
						padding=layer_pad_2,
						kernel_size=(kernel_size_2, kernel_size_2))
		self.act2 = ReLU()
		# self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize first (and only) set of FC => RELU layers
		fc1_features = layer_width_2 * layer_width_2 * out_channels_2
		fc2_features = parameters.fc2
		fc3_features = parameters.fc3
		self.fc1 = Linear(in_features=fc1_features, out_features=fc2_features)
		self.act3 = ReLU()
		self.fc2 = Linear(in_features=fc2_features, out_features=fc3_features)
		self.act4 = ReLU()
		self.fc3 = Linear(in_features=fc3_features, out_features=1)


	# x will be 7x7x5, onehot encoded
	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.act1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.act2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.act3(x)
		x = self.fc2(x)
		x = self.act4(x)
		x = self.fc3(x)
		x = flatten(x)
		return x