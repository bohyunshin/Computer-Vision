### Reference

* Deep Residual Learning for Image Recognition, Kaiming He et al.

### Intro

굉장히 깊은 신경망은 좋은 성능을 보일 것으로 기대된다. 그러나 좋은 점이 있으면 나쁜 점도 있는 법. 이렇게 깊은 신경망은 훈련하기가 어렵고 설사 가능하다고 하더라도 시간이 오래 걸린다. Resnet은 이러한 단점을 보완한다. layers의 입력으로 residual을 배우게 함으로써 더 나은 결과를 유도하고 쉽게 최적화하며 계산량을 줄인다. 이는 곧, ImageNet 데이터 세트에 대해서, VGG net보다 8배 깊지만 계산량은 작다는 사실로부터 증명된다.

깊은 신경망이 가지는 문제를 degradation라고 부른다. 이는 layer가 깊어질수록 정확도는 어느 순간 상승하지 않고 빠르게 감소한다는 것이다. *이는 훈련 데이터에 대한 과적합 현상이 아니다.* 이는 아래 실험 결과로부터도 나타난 현상이다.

<img src="C:\Users\sbh0613\AppData\Roaming\Typora\typora-user-images\image-20200218140607037.png" alt="image-20200218140607037" style="zoom:67%;" />

왼쪽 그림은 20-layer 신경망이고 오른쪽 그림은 56-layer 신경망이다. 더 깊은 신경망의 훈련, 테스트 에러가 더 얕은 신경망의 그것들보다 더 높음을 확인할 수 있다.

해당 논문에서 제시하는 ResNet은 deep residual learning으로써, 이러한 degradation 문제를 해결한다. 먼저 residual learning의 도식을 아래 그림으로 살펴보자. 

<img src="C:\Users\sbh0613\AppData\Roaming\Typora\typora-user-images\image-20200218143143343.png" alt="image-20200218143143343" style="zoom:67%;" />

아래와 같이 notation을 정의하자.

* $\mathcal{H(x)}$ : desired underlying mapping **[Unreferenced mapping, Underlying mapping]** 
*  $\mathcal{F(x)}:=\mathcal{H(x)}-x$ : another mapping which stacked nonlinear layers fit **[Residual mapping]** 
* $\mathcal{F(x)+x}$ : original fitting which is recast
* $x$ : input

원래의 unreferenced mapping보다는, residual mapping이 최적화하기 쉬움을 가정한다.
$\mathcal{F(x)+x}$ 은 위 그림과 같은 *shortcut connections*으로 구현할 수 있다. *shortcut connection*은 한개 이상의 layers을 skip하는 것이다. 위 그림에서 shortcut connections는 *identity* mapping이고 이 mapping의 결과물이 stacked layers의 출력물에 더해진다. 이러한 identity mapping은 더 많은 모수나 computational complexity을 필요로하지 않는다.

### Deep Residual Learning

#### 1. Residual Learning

앞서 $\mathcal{H(x)}$을 underlying mapping이라 했고 이는 우리가 세운 가설(hypothesis)로써, 몇 개의 layers을 통해 알아내고 싶은 대상이다. 그런데 residual learning에서는 stacked layers로 하여금 $\mathcal{H(x)}$가 아니라 residual function인 $\mathcal{F(x)}:=\mathcal{H(x)}-x$ 을 근사하게 한다. 따라서 원래의 가설은 $\mathcal{H(x)}:=\mathcal{F(x)}+x$이 된다.
이러한 재구성은 degradation 때문에 하는 것이다. 앞서 언급한 degradation은 모델에 identity mapping이 추가된다면 더 깊은 모델은 얕은 모델보다 좋은 성능을 가져야 할 것으로 기대되지만 여러 개의 nonlinear layers로 인해서 identity mapping을 학습하기 쉽지 않은 현상이다. residual learning의 재구성으로, 만약에 identity mapping이 적절하다면, solvers는 nonlinear layers의 weights을 0로 만들어서 identity mapping과 가깝게 만들 것이다.
하지만 실제에서 identity mapping이 적절한 경우는 많지 않다. 그럼에도 이러한 재구성은 문제를 precondition하는데 도움을 준다. solvers가 풀어야할 optimal function이 zero mapping이 아니라 identity function에 가깝다면 solver 입장에서는 아예 새로운 것을 배우는 것보다 identity mapping에서 작은 변화를 찾는 것이 더 쉬울 것이다.

#### 2. Identity Mapping by Shortcuts

몇개의 stacked layers마다 residual learning을 도입한다. 본 논문에서 building block을 아래와 같이 정의한다.
$$
y=\mathcal{F}(x,W_i)\;+\;x​\tag{1}
$$
여기서 $x$ 와 $y$는 각각 layers의 입, 출력 벡터이다. 함수 $\mathcal{F}$는 학습되어야하는 residual mapping이다. 예를 들어서 두 개의 layers를 가정할 때, $\mathcal{F}=W_2\sigma(W_1x)$이다 (여기서 $\sigma$는 ReLU 함수) 두 개의 layers를 가질 때, 이와 같이 $\mathcal{F}$를 정의하고 $\mathcal{F}+x$를 통해 shortcut connection을 수행한다. 그리고 그 후 두 번째 nonlinearity을 적용한다 (위 그림에서 shortcut 이후에 ReLU가 한번 더 있다.)
$x$와 $\mathcal{F}$는 동일한 차원을 가져야 한다. 만약 그렇지 않다면 linear projection $W_s$을 통해서 차원을 맞춰야 한다.
$$
y=\mathcal{F}(x,W_i)\;+\;W_sx​\tag{2}
$$
$\mathcal{F}$의 형태는 제약이 없지만 한 개의 layers만 가진다면, 딱히 의미가 없는 그저 linear layer로 될 것이다. 즉, $y=W_1x + x$ 가 된다. 또한 굳이 FC가 아니더라도 CNN이 될 수도 있다.

#### 3. Architecture

<img src="C:\Users\sbh0613\AppData\Roaming\Typora\typora-user-images\image-20200218160825267.png" alt="image-20200218160825267" style="zoom:67%;" />

위 그름은 왼쪽부터 차례대로 VGG, plain network, residual network의 아키텍처이다. residual network을 사용한 아키텍처는 옆쪽에 화살표로 표시된 shortcut connections이 존재한다.