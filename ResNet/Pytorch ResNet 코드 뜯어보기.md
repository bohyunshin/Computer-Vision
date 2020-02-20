# Pytorch ResNet 코드 뜯어보기

목표: pytorch에 구현되어 있는 ResNet 코드를 뜯어보며 깊게 이해해보자.

### Reference

* https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
* https://medium.com/@erikgaas/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem-145620f93096



```python 
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
```

이 함수는 nn.Conv2d를 반환한다. kernel_size가 3이고 pading, stride가 모두 1인 것으로 보아, 차원을 유지하는 convolution임을 알 수 있다. 필터의 크기 때문에 $3\times3$이라고 이름 붙인것 같다.

```python 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
```

이 함수도 nn.Conv2d를 반환하는데 kernel_size가 1이다. padding은 기본값으로 0이고 stride는 1이다.

#### BasicBlock

이제 BasicBlock 클래스를 선언한다.

```python 
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
```

pytorch에서 클래스를 선언할 때와 동일하게 nn.Module을 상속받고 ```super(BasicBlock, self).__init__()```을 통해 ```nn.Modeul.__init(self)__``` 을 발동시킨다.
```BasicBlock``` 클래스는 인자로 inplanes, planes를 받는다. inplanes는 첫 번째 ```conv1```의 input channel이고 planes은 output channel이다. 그리고 이 planes가 다음 ```conv2``` 때에는 유지가 된다. 즉 ```conv2```의 input channel과 output channel은 동일하다.
norm_layer 인자를 통해 BatchNorm을 수행한다. 이후 self을 정의한 부분을 보면,  ```conv, bn, relu```을 차례로 정의한다. 

```python 
# class BasicBlock(nn.Module)
def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

다음으로 ```BasicBlock```  클래스가 호출되면 자동으로 수행되는 ```forward``` 함수이다. 순서대로 ```conv1, bn1, relu, conv2, bn2```가 수행되고 ```identity```를 더해준다. 그리고 ```relu```에 통과시킨다. 이는 블록을 코드로 구현한 것이다.

<img src="https://user-images.githubusercontent.com/36855000/74916058-88370480-5408-11ea-99dd-36906add2a0f.png">

ResNet 논문에서는 weight layer에 FC든, CNN이든 어떠한 것이 올 수 있다고 말한다. 여기서는 weight layer로 CNN이 사용된 것이다. 또한 두 개의 CNN층을 통과한 output에 ```identity```를 더하고 여기에 ```relu```을 적용하는 모습은 위 그림과 정확히 일치한다.

#### Bottleneck

다음으로 Bottleneck 클래스이다.

```python 
class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
```

```BasicBlock```과 마찬가지로 inplanes, planes를 인자로 받는다.
우선 inplanes는 첫 번째 ```conv1```의 input channel이다. 첫 번째 convolution인 `conv1`는 inplanes와 width을 인자로 받는다.  width의 정의를 살펴보면, 만약 `Bottleneck`의 기본값 인자를 사용한다면 그저 planes 값을 가진다. 어쨌든, input channels로 inplanes을, output channels로 width을 받는다.
두 번째 `conv2`는 input channels와 output channels을 동일하게 width로 받는다.
세 번째 `conv3`는 output channels에 `self.expansion`을 곱하여 차원을 증가시킨다.

그렇다면 마지막 `conv3`에서 차원을 증가시킬까? 앞서 `conv1, conv2`을 거치며 차원이 감소하기 때문에 이를 원래대로 맞춰야하기 때문이다.
왜 원래대로 맞춰야할까? `Bottleneck` 블록을 통해 layer의 output과 identity을 더해야하기 때문이다.

다음으로 `Bottleneck`의 `forward` 함수를 보자.

```python
# class Bottleneck(nn.Module):
def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

이는 앞의 `BasicBlock` 클래스의 `forward` 함수와 convolution layer가 세 개라는 점을 제외하고는 동일한 구조이다.

그렇다면 `BasicBlock`과 `Bottleneck`의 차이점은 무엇일까? [여기](https://medium.com/@erikgaas/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem-145620f93096)의 자세한 설명을 살펴보자.

<img src="https://user-images.githubusercontent.com/36855000/74916056-879e6e00-5408-11ea-976e-0f024a1c0a83.png">

왼쪽이 `BasicBlcok`, 오른쪽이 `Bottleneck`이다. `BasicBlcok`은 $3\times3$ convolution을 사용하는데, 이게 layers가 깊어질수록 계산량이 장난 아니라고 한다. 반면에 `Bottleneck`과 같이 $1\times1$ convolution을 시작과 끝에 해주면 계산량은 훨씬 줄일 수 있다고 한다. 대신, 차원이 줄어드니까 expansion을 하는 것이다.

#### Resnet

다음으로는 ResNet 클래스이다. 먼저 ResNet의 전체적인 아키텍쳐를 살펴보자.
논문에 따르면 image를 입력 받고 거치는 첫 번째 `conv1`은 아래와 같다.

<img src = "https://user-images.githubusercontent.com/36855000/74916054-879e6e00-5408-11ea-8f3a-f04ca90ae1fb.png">

즉, 필터의 크기가 $7\times7$이며 output channels은 64이다. 이를 코드로 구현하면 아래와 같다.

```python
self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
```

nn.Conv2d의 input channel이 3인데, 그 이유는 입력으로 받는 이미지의 채널의 크기가 3이기 때문이다.
layer에 대한 코드를 이해하기 위해서는 우선 아래의 함수를 음미해야 한다.

```python
 def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
```

함수의 인자를 살펴보자. block은 `BasicBlock`이나 `Bottleneck`과 같은 block을 뜻한다. planes는 out channels을 뜻하고 blocks는 block의 갯수이다.
if문을 살펴보자. `stride != 1 or self.inplanes != planes * block.expansion`이라면 `downsample`을 하는데 여기서 downsample은 차원을 맞춰주는 것으로 이해하자. 그런데, `self.inplanes`부터 우선 이해를 해야할 것 같다. ResNet 클래스 초반부를 살펴보자.

```python
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
```

`self.inplanes = 64`라고 나와 있다. 그렇다면 input channels이 64라는 뜻인데, 앞서 필터가 7인 `conv1`에서 output channels이 64고, 이것이 바로 다름 layer의 input channels로 들어가기 때문에 `self.inplanes = 64`로 고정해둔 것이다.

다시 `stride != 1 or self.inplanes != planes * block.expansion`을 살펴보자. `block.expansion`은 `BasicBlock`이라면 1이고 `Bottleneck`이라면 4이다. `BasicBlock`은 kernel의 크기가 3, padding, stride가 1인 convolution이므로 크기를 보존한다. 따라서 input, otuput channel이 동일하다. 하지만 `Bottleneck`은 output channel의 크기가 다르기 때문에 `block.expansion`만큼 크기를 보정해줘야 한다. 이래야 identity와 더하는 residual net을 구현할 수 있기 때문이다.

```python
# 위의 _make_layer() 함수의 continue
def _make_layer():
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
```

위 코드에서 block()이 의미하는 바를 생각해야 한다. block은 ResNet 클래스의 인자이다. 여기서는 `BasicBlock` 또는 `Bottleneck`이다. 따라서 layers에 둘 중 하나 블록을 추가한다는 뜻이다.

```python
# _make_layer() 함수의 continue
	self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer))

        return nn.Sequential(*layers)
```

최종으로 인자로 받은 blocks의 크기만큼 for 문을 돌며 layers을 쌓는다. 마지막으로 쌓은 layers을 nn.Sequential()로 묶어서 반환한다.

_make_layer()에 대해서 충분히 음미하였으니 이제 아래를 살펴보자.

```python
self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
self.bn1 = norm_layer(self.inplanes)
self.relu = nn.ReLU(inplace=True)
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
self.layer1 = self._make_layer(block, 64, layers[0])
self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                               dilate=replace_stride_with_dilation[0])
self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                               dilate=replace_stride_with_dilation[1])
self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                               dilate=replace_stride_with_dilation[2])
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.fc = nn.Linear(512 * block.expansion, num_classes)
```

나머지는 쉽게 알 수 있고 `self.layer`만 살펴보자. _make_layer 함수를 이용해, 사용할 block, out channels의 크기를 받는다. 또한 리스트 layers는 ResNet 클래스의 인자이다. 즉, 각 layer에서 몇 개를 쌓을지 입력받는다. 따라서 `layer[0]`은 첫 번째 layer의 층 개수라는 뜻이다.

`layer1`부터 `layer4`까지는 논문의 resnet 아키텍처에서 네 종류의 layer와 대응된다. 예를 들어 `layer1`은 아래 아키텍처와 대응된다.

<img width="400" src="https://user-images.githubusercontent.com/36855000/74916052-866d4100-5408-11ea-872b-de5f3e0a6577.png">



 이 과정을 거치고 pooling을 하고 마지막으로는 FC을 거쳐 1000개의 클래스에 대한 점수를 반환한다.
이를 forward 함수로 구현한 코드는 아래와 같다.

```python
def _forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

def forward(self, x):
    return self._forward_impl(x)
```

논문의 아키텍처와 정확히 일치함을 알 수 있다.

이상으로 pytorch에서 구현된 ResNet 코드를 뜯어보았다. 생각보다 그렇게 어렵지 않았고, 확실히 한줄 한줄 뜯어보니 논문의 내용에 대한 복습도 되고 이해도 잘 되는 것 같았다. 다음에는 kaggle의 이미지 분류 문제에 pytorch을 이용하여 간단히 resnet을 적용해보겠다.
