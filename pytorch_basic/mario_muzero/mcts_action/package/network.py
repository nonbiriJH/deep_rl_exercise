from typing import Dict, List, NamedTuple
from action import Action
from torch import nn

class ResNet(nn.Module):
    def __init__(self, n_filter, n_padding):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=n_filter, out_channels = n_filter, kernel_size = 3,padding=n_padding),
            nn.BatchNorm2d(n_filter),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_filter, out_channels = n_filter, kernel_size = 3,padding=n_padding),
            nn.BatchNorm2d(n_filter),
          )

    def forward(self, inputs):
        return nn.ReLU()(self.module(inputs) + inputs)

class MuNet(nn.Module):

    def __init__(self, policy_dim, is_reference):
        super().__init__()
        self.policy_dim = policy_dim
        self.is_reference = is_reference
        if is_reference:
          self.body_feature = 256
        else:
          self.body_feature = 256 + self.policy_dim
        
        #INPUT NET
        #Representation - input observation 96x96x(32x(3+1))
        self.rep = nn.Sequential(
          nn.Conv2d(128,128,3,stride=2, padding=1), #48x48x128
          ResNet(128,1),
          ResNet(128,1),
          nn.Conv2d(128,256,3,stride=2, padding=1), #24x24x256
          ResNet(256,1),
          ResNet(256,1),
          ResNet(256,1),
          nn.AvgPool2d(2), #12x12x256
          ResNet(256,1),
          ResNet(256,1),
          ResNet(256,1),
          nn.AvgPool2d(2),#6x6x256 hidden state out
        )
        #Dynamic - input hidden state 6x6x256 and action 6x6xpolicy_dim

        #OUTPUT NET
        self.pol = nn.Sequential(
          nn.Conv2d(in_channels=self.body_feature, out_channels = 1, kernel_size = 1), #2 conv layer
          nn.BatchNorm2d(1),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(in_features = 6*6,out_features=self.policy_dim),
        )

        self.val = nn.Sequential(
          
          nn.Conv2d(in_channels=self.body_feature, out_channels = 1, kernel_size = 1), #1 conv layer
          nn.BatchNorm2d(1),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(in_features = 6*6,out_features=6),
          nn.ReLU(),
          nn.Linear(in_features =6,out_features=1),
          #tanh
        )

        #BODY NET
        self.body = nn.Sequential()
        for _ in range(16):
          self.body = nn.Sequential(
            self.body,
            ResNet(self.body_feature,1))

    def forward(self, input):
      if self.is_reference:
        net = nn.Sequential(self.rep,self.body)
        out = net(input)
      return self.pol(out),self.val(out)

class NetworkOutput(NamedTuple):
  value: float
  reward: float
  policy_logits: Dict[Action, float]
  hidden_state: List[float]

class Network(object):

  def initial_inference(self, image) -> NetworkOutput:
    # representation + prediction function
    return NetworkOutput(0, 0, {}, [])

  def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
    # dynamics + prediction function
    return NetworkOutput(0, 0, {}, [])

  def get_weights(self):
    # Returns the weights of this network.
    return []

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0

  def make_uniform_network():
    return Network()

#store network output
class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return Network.make_uniform_network()

  def save_network(self, step: int, network: Network):
    self._networks[step] = network

