---
layout: page
permalink: /convolutional-networks-kr/
---

Table of Contents:

- [Architecture Overview](#overview)
- [ConvNet Layers](#layers)
  - [Convolutional Layer](#conv)
  - [Pooling Layer](#pool)
  - [Normalization Layer](#norm)
  - [Fully-Connected Layer](#fc)
  - [Converting Fully-Connected Layers to Convolutional Layers](#convert)
- [ConvNet Architectures](#architectures)
  - [Layer Patterns](#layerpat)
  - [Layer Sizing Patterns](#layersizepat)
  - [Case Studies](#case) (LeNet / AlexNet / ZFNet / GoogLeNet / VGGNet)
  - [Computational Considerations](#comp)
- [Additional References](#add)

## 컨볼루션 신경망 (CNN/ConvNets)

컨볼루션 신경망 (Convolutional Neural Network, 이하 CNN)은 앞 장에서 다룬 일반 신경망과 매우 유사하다. CNN은 학습 가능한 가중치 (weight)와 바이어스(bias)로 구성되어 있다. 각 뉴런은 입력을 받아 내적 연산( dot product )을 한 뒤 선택에 따라 비선형 (non-linear) 연산을 한다. 전체 네트워크는 일반 신경망과 마찬가지로 미분 가능한 하나의 스코어 함수 (score function)을 갖게 된다 (맨 앞쪽에서 로우 이미지 (raw image)를 읽고 맨 뒤쪽에서 각 클래스에 대한 점수를 구하게 됨). 또한 CNN은 마지막 레이어에 (SVM/Softmax와 같은) 손실 함수 (loss function)을 가지며, 우리가 일반 신경망을 학습시킬 때 사용하던 각종 기법들을 동일하게 적용할 수 있다.

CNN과 일반 신경망의 차이점은 무엇일까? CNN 아키텍쳐는 입력 데이터가 이미지라는 가정 덕분에 이미지 데이터가 갖는 특성들을 인코딩 할 수 있다. 이러한 아키텍쳐는 포워드 함수 (forward function)을 더욱 효과적으로 구현할 수 있고 네트워크를 학습시키는데 필요한 모수 (parameter)의 수를 크게 줄일 수 있게 해준다.

<a name='overview'></a>

### 아키텍쳐 개요

앞 장에서 보았듯이 신경망은 입력받은 벡터를 일련의 히든 레이어 (hidden layer) 를 통해 변형 (transform) 시킨다. 각 히든 레이어는 뉴런들로 이뤄져 있으며, 각 뉴런은 앞쪽 레이어 (previous layer)의 모든 뉴런과 연결되어 있다 (fully connected). 같은 레이어 내에 있는 뉴런들 끼리는 연결이 존재하지 않고 서로 독립적이다. 마지막 Fully-connected 레이어는 출력 레이어라고 불리며, 분류 문제에서 클래스 점수 (class score)를 나타낸다.

일반 신경망은 이미지를 다루기에 적절하지 않다. CIFAR-10 데이터의 경우 각 이미지가 32x32x3 (가로,세로 32, 3개 컬러 채널)로 이뤄져 있어서 첫 번째 히든 레이어 내의 하나의 뉴런의 경우 32x32x3=3072개의 가중치가 필요하지만, 더 큰 이미지를 사용할 경우에는 같은 구조를 이용하는 것이 불가능하다. 예를 들어 200x200x3의 크기를 가진 이미지는 같은 뉴런에 대해 200x200x3=120,000개의 가중치를 필요로 하기 때문이다. 더욱이, 이런 뉴런이 레이어 내에 여러개 존재하므로 모수의 개수가 크게 증가하게 된다. 이와 같이 Fully-connectivity는 심한 낭비이며 많은 수의 모수는 곧 오버피팅(overfitting)으로 귀결된다.

CNN은 입력이 이미지로 이뤄져 있다는 특징을 살려 좀 더 합리적인 방향으로 아키텍쳐를 구성할 수 있다. 특히 일반 신경망과 달리, CNN의 레이어들은 가로,세로,깊이의 3개 차원을 갖게 된다 ( 여기에서 말하는 깊이란 전체 신경망의 깊이가 아니라 액티베이션 볼륨 ( activation volume ) 에서의 3번 째 차원을 이야기 함 ). 예를 들어 CIFAR-10 이미지는 32x32x3 (가로,세로,깊이) 의 차원을 갖는 입력 액티베이션 볼륨 (activation volume)이라고 볼 수 있다. 조만간 보겠지만, 하나의 레이어에 위치한 뉴런들은 일반 신경망과는 달리 앞 레이어의 전체 뉴런이 아닌 일부에만 연결이 되어 있다. CNN 아키텍쳐는 전체 이미지를 클래스 점수들로 이뤄진 하나의 벡터로 만들어주기 때문에 마지막 출력 레이어는 1x1x10(10은 CIFAR-10 데이터의 클래스 개수)의 차원을 가지게 된다. 이에 대한 그럼은 아래와 같다:

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/nn1/neural_net2.jpeg" width="40%">
  <img src="{{site.baseurl}}/assets/cnn/cnn.jpeg" width="48%" style="border-left: 1px solid black;">
  <div class="figcaption">좌: 일반 3-레이어 신경망. 우: 그림과 같이 CNN은 뉴런들을 3차원으로 배치한다. CNN의 모든 레이어는 3차원 입력 볼륨을 3차원 출력 볼륨으로 변환 (transform) 시킨다. 이 예제에서 붉은 색으로 나타난 입력 레이어는 이미지를 입력으로 받으므로, 이 레이어의 가로/세로/채널은 각각 이미지의 가로/세로/3(Red,Green,Blue) 이다.</div>
</div>

> CNN은 여러 레이어로 이루어져 있다. 각각의 레이어는 3차원의 볼륨을 입력으로 받고 미분 가능한 함수를 거쳐 3차원의 볼륨을 출력하는  간단한 기능을 한다.

<a name='layers'></a>

### CNN을 이루는 레이어들

위에서 다룬 것과 같이, CNN의 각 레이어는 미분 가능한 변환 함수를 통해 하나의 액티베이션 볼륨을 또다른 액티베이션 볼륨으로 변환 (transform) 시킨다. CNN 아키텍쳐에서는 크게 컨볼루셔널 레이어, 풀링 레이어, Fully-connected 레이어라는 3개 종류의 레이어가 사용된다. 전체 CNN 아키텍쳐는 이 3 종류의 레이어들을 쌓아 만들어진다.

*예제: 아래에서 더 자세하게 배우겠지만, CIFAR-10 데이터를 다루기 위한 간단한 CNN은 [INPUT-CONV-RELU-POOL-FC]로 구축할 수 있다.

- INPUT 입력 이미지가 가로32, 세로32, 그리고 RGB 채널을 가지는 경우 입력의 크기는 [32x32x3].
- CONV 레이어는 입력 이미지의 일부 영역과 연결되어 있으며, 이 연결된 영역과 자신의 가중치의 내적 연산 (dot product) 을 계산하게 된다. 결과 볼륨은 [32x32x12]와 같은 크기를 갖게 된다.
- RELU 레이어는 max(0,x)와 같이 각 요소에 적용되는 액티베이션 함수 (activation function)이다. 이 레이어는 볼륨의 크기를 변화시키지 않는다 ([32x32x12])
- POOL 레이어는 (가로,세로) 차원에 대해 다운샘플링 (downsampling)을 수행해 [16x16x12]와 같이 줄어든 볼륨을 출력한다.
- FC (fully-connected) 레이어는 클래스 점수들을 계산해 [1x1x10]의 크기를 갖는 볼륨을 출력한다. 10개 숫자들은 10개 카테고리에 대한 클래스 점수에 해당한다. 레이어의 이름에서 유추 가능하듯, 이 레이어는 이전 볼륨의 모든 요소와 연결되어 있다.

이와 같이, CNN은 픽셀 값으로 이뤄진 원본 이미지를 각 레이어를 거치며 클래스 점수로 변환 (transform) 시킨다. 한 가지 기억할 것은, 어떤 레이어는 모수 (parameter)를 갖지만 어떤 레이어는 모수를 갖지 않는다는 것이다. 특히 CONV/FC 레이어들은 단순히 입력 볼륨만이 아니라 가중치(weight)와 바이어스(bias) 또한 포함하는 액티베이션(activation) 함수이다. 반면 RELU/POOL 레이어들은 고정된 함수이다. CONV/FC 레이어의 모수 (parameter)들은 각 이미지에 대한 클래스 점수가 해당 이미지의 레이블과 같아지도록 그라디언트 디센트 (gradient descent)로 학습된다.

요약해보면:

- CNN 아키텍쳐는 여러 레이어를 통해 입력 이미지 볼륨을 출력 볼륨 ( 클래스 점수 )으로 변환시켜 준다.
- CNN은 몇 가지 종류의 레이어로 구성되어 있다. CONV/FC/RELU/POOL 레이어가 현재 가장 많이 쓰인다.
- 각 레이어는 3차원의 입력 볼륨을 미분 가능한 함수를 통해 3차원 출력 볼륨으로 변환시킨다.
- 모수(parameter)가 있는 레이어도 있고 그렇지 않은 레이어도 있다 (FC/CONV는 모수를 갖고 있고, RELU/POOL 등은 모수가 없음).
- 초모수 (hyperparameter)가 있는 레이어도 있고 그렇지 않은 레이어도 있다 (CONV/FC/POOL 레이어는 초모수를 가지며 RELU는 가지지 않음).

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/cnn/convnet.jpeg" width="100%">
  <div class="figcaption">
    CNN 아키텍쳐의 액티베이션 (activation) 예제. 첫 볼륨은 로우 이미지(raw image)를 다루며, 마지막 볼륨은 클래스 점수들을 출력한다. 입/출력 사이의 액티베이션들은 그림의 각 열에 나타나 있다. 3차원 볼륨을 시각적으로 나타내기가 어렵기 때문에 각 행마다 볼륨들의 일부만 나타냈다. 마지막 레이어는 모든 클래스에 대한 점수를 나타내지만 여기에서는 상위 5개 클래스에 대한 점수와 레이블만 표시했다. <a href="http://cs231n.stanford.edu/">전체 웹 데모</a>는 우리의 웹사이트 상단에 있다. 여기에서 사용된 아키텍쳐는 작은 VGG Net이다.
  </div>
</div>

이제 각각의 레이어에 대해 초모수(hyperparameter)나 연결성 (connectivity) 등의 세부 사항들을 알아보도록 하자.

<a name='conv'></a>

#### 컨볼루셔널 레이어 (이하 CONV)

CONV 레이어는 CNN을 이루는 핵심 요소이다. CONV 레이어의 출력은 3차원으로 정렬된 뉴런들로 해석될 수 있다. 이제부터는 뉴런들의 연결성 (connectivity), 그들의 공간상의 배치, 그리고 모수 공유(parameter sharing) 에 대해 알아보자.

**개요 및 직관적인 설명.** CONV 레이어의 모수(parameter)들은 일련의 학습가능한 필터들로 이뤄져 있다. 각 필터는 가로/세로 차원으로는 작지만 깊이 (depth) 차원으로는 전체 깊이를 아우른다. 포워드 패스 (forward pass) 때에는 각 필터를 입력 볼륨의 가로/세로 차원으로 슬라이딩 시키며 (정확히는 convolve 시키며) 2차원의 액티베이션 맵 (activation map)을 생성한다. 필터를 입력 위로 슬라이딩 시킬 때, 필터와 입력의 요소들 사이의 내적 연산 (dot product)이 이뤄진다. 직관적으로 설명하면, 이 신경망은 입력의 특정 위치의 특정 패턴에 대해 반응하는 (activate) 필터를 학습한다.  이런 액티베이션 맵 (activation map)을 깊이 (depth) 차원을 따라 쌓은 것이 곧 출력 볼륨이 된다. 그러므로 출력 볼륨의 각 요소들은 입력의 작은 영역만을 취급하고, 같은 액티베이션 맵 내의 뉴런들은 같은 모수들을 공유한다 (같은 필터를 적용한 결과이므로).  이제 이 과정에 대해 좀 더 깊이 파헤쳐보자.

**로컬 연결성 (Local connectivity).** 이미지와 같은 고차원 입력을 다룰 때에는, 현재 레이어의 한 뉴런을 이전 볼륨의 모든 뉴런들과 연결하는 것이 비 실용적이다. 대신에 우리는 레이어의 각 뉴런을 입력 볼륨의 로컬한 영역(local region)에만 연결할 것이다. 이 영역은 리셉티브 필드 (receptive field)라고 불리는 초모수 (hyperparameter) 이다. 깊이 차원 측면에서는 항상 입력 볼륨의 총 깊이를 다룬다 (가로/세로는 작은 영역을 보지만 깊이는 전체를 본다는 뜻). 공간적 차원 (가로/세로)와 깊이 차원을 다루는 방식이 다르다는 걸 기억하자.

*예제 1*. 예를 들어 입력 볼륨의 크기가 (CIFAR-10의 RGB 이미지와 같이) [32x32x3]이라고 하자. 만약 리셉티브 필드의 크기가 5x5라면, CONV 레이어의 각 뉴런은 입력 볼륨의 [5x5x3] 크기의 영역에 가중치 (weight)를 가하게 된다 (총 5x5x3=75 개 가중치). 입력 볼륨 (RGB 이미지)의 깊이가 3이므로 마지막 숫자가 3이 된다는 것을 기억하자.

*예제 2*. 입력 볼륨의 크기가 [16x16x20]이라고 하자. 3x3 크기의 리셉티브 필드를 사용하면 CONV 레이어의 각 뉴런은 입력 볼륨과 3x3x20=180 개의 연결을 갖게 된다. 이번에도 입력 볼륨의 깊이가 20이므로 마지막 숫자가 20이 된다는 것을 기억하자.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/cnn/depthcol.jpeg" width="40%">
  <img src="{{site.baseurl}}/assets/nn1/neuron_model.jpeg" width="40%" style="border-left: 1px solid black;">
  <div class="figcaption">
    <b>좌:</b> 입력 볼륨(붉은색, 32x32x3 크기의 CIFAR-10 이미지)과 첫번째 컨볼루션 레이어 볼륨. 컨볼루션 레이어의 각 뉴런은 입력 볼륨의 일부 영역에만 연결된다 (가로/세로 공간 차원으로는 일부 연결, 깊이(컬러 채널) 차원은 모두 연결). 컨볼루션 레이어의 깊이 차원의 여러 뉴런 (그림에서 5개)들이 모두 입력의 같은 영역을 처리한다는 것을 기억하자 (깊이 차원과 관련해서는 아래에서 더 자세히 알아볼 것임). 우: 입력의 일부 영역에만 연결된다는 점을 제외하고는, 이전 신경망 챕터에서 다뤄지던 뉴런들과 똑같이 내적 연산과 비선형 함수로 이뤄진다.
  </div>
</div>

**공간적 배치**. 지금까지는 컨볼루션 레이어의 한 뉴런과 입력 볼륨의 연결에 대해 알아보았다. 그러나 아직 출력 볼륨에 얼마나 많은 뉴런들이 있는지, 그리고 그 뉴런들이 어떤식으로 배치되는지는 다루지 않았다. 3개의 hyperparameter들이 출력 볼륨의 크기를 결정하게 된다. 그 3개 요소는 바로 **깊이, stride, 그리고 제로 패딩 (zero-padding)** 이다. 이들에 대해 알아보자:

1. 먼저, 출력 볼륨의 **깊이** 는 우리가 결정할 수 있는 요소이다. 컨볼루션 레이어의 뉴런들 중 입력 볼륨 내 동일한 영역과 연결된 뉴런의 개수를 의미한다. 마치 일반 신경망에서 히든 레이어 내의 모든 뉴런들이 같은 입력값과 연결된 것과 비슷하다. 앞으로 살펴보겠지만, 이 뉴런들은 입력에 대해 서로 다른 특징 (feature)에 활성화된다 (activate). 예를 들어, 이미지를 입력으로 받는 첫 번째 컨볼루션 레이어의 경우, 깊이 축에 따른 각 뉴런들은 이미지의 서로 다른 엣지, 색깔, 블롭(blob) 등에 활성화된다. 앞으로는 인풋의 서로 같은 영역을 바라보는 뉴런들을 **깊이 컬럼 (depth column)**이라고 부르겠다.
2. 두 번째로 어떤 간격 (가로/세로의 공간적 간격) 으로 깊이 컬럼을 할당할 지를 의미하는 **stride**를 결정해야 한다. 만약 stride가 1이라면, 깊이 컬럼을 1칸마다 할당하게 된다 (한 칸 간격으로 깊이 컬럼 할당). 이럴 경우 각 깊이 컬럼들은 receptive field 상 넓은 영역이 겹치게 되고, 출력 볼륨의 크기도 매우 커지게 된다. 반대로, 큰 stride를 사용한다면 receptive field끼리 좁은 영역만 겹치게 되고 출력 볼륨도 작아지게 된다 (깊이는 작아지지 않고 가로/세로만 작아지게 됨).
3. 조만간 살펴보겠지만, 입력 볼륨의 가장자리를 0으로 패딩하는 것이 좋을 때가 있다. 이 **zero-padding**은 hyperparamter이다. zero-padding을 사용할 때의 장점은, 출력 볼륨의 공간적 크기(가로/세로)를 조절할 수 있다는 것이다. 특히 입력 볼륨의 공간적 크기를 유지하고 싶은 경우 (입력의 가로/세로 = 출력의 가로/세로) 사용하게 된다.

출력 볼륨의 공간적 크기 (가로/세로)는 입력 볼륨 크기 ($$W$$), CONV 레이어의 리셉티브 필드 크기($$F$$)와 stride ($$S$$), 그리고 제로 패딩 (zero-padding) 사이즈 ($$P$$) 의 함수로 계산할 수 있다. $$(W - F + 2P)/S + 1$$. I을 통해 알맞은 크기를 계산하면 된다. 만약 이 값이 정수가 아니라면 stride가 잘못 정해진 것이다. 이 경우 뉴런들이 대칭을 이루며 깔끔하게 배치되는 것이 불가능하다. 다음 예제를 보면 이 수식을 좀 더 직관적으로 이해할 수 있을 것이다:

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/cnn/stride.jpeg">
  <div class="figcaption">
  공간적 배치에 관한 그림. 이 예제에서는 가로/세로 공간적 차원 중 하나만 고려한다 (x축). 리셉티브 필드 F=3, 입력 사이즈 W=5, 제로 패딩 P=1. <b>좌</b>: 뉴런들이 stride S=1을 갖고 배치된 경우,  출력 사이즈는 (5-3+2)/1 +1 = 5이다. <b>우</b>: stride S=2인 경우 (5-3+2)/2 + 1 = 3의 출력 사이즈를 가진다. Stride S=3은 사용할 수 없다. (5-3+2) = 4가 3으로 나눠지지 않기 때문에 출력 볼륨의 뉴런들이 깔끔히 배치되지 않는다.
  이 예에서 뉴런들의 가중치는 [1,0,-1] (가장 오른쪽) 이며 bias는 0이다. 이 가중치는 노란 뉴런들 모두에게 공유된다 (아래에서 parameter sharing에 대해 살펴보라).
  </div>
</div>

*제로 패딩 사용*. 위 예제의 왼쪽 그림에서, 입력과 출력의 차원이 모두 5라는 것을 기억하자. 리셉티브 필드가 3이고 제로 패딩이 1이기 때문에 이런 결과가 나오는 것이다. 만약 제로 패딩이 사용되지 않았다면 출력 볼륨의 크기는 3이 될 것이다. 일반적으로, 제로 패딩을 $$P = (F - 1)/2$$ , stride $$S = 1$$로 세팅하면 입/출력의 크기가 같아지게 된다. 이런 방식으로 사용하는 것이 일반적이며, 앞으로 컨볼루션 신경망에 대해 다루면서 그 이유에 대해 더 알아볼 것이다.

*Stride에 대한 constraints*. 공간적 배치와 관련된 hyperparameter들은 상호 constraint들이 존재한다는 것을 기억하자. 예를 들어, 입력 사이즈 $$W=10$$이고 제로 패딩이 사용되지 않았고 $$P=0$$, 필터 사이즈가 $$F=3$$이라면, stride $$S=2$$를 사용하는 것이 불가능하다. $$(W - F + 2P)/S + 1 = (10 - 3 + 0) / 2 + 1 = 4.5$$이 정수가 아니기 때문이다. 그러므로 hyperparameter를 이런 식으로 설정하면 컨볼루션 신경망 관련 라이브러리들은 exception을 낸다. 컨볼루션 신경망의 구조 관련 섹션에서 확인하겠지만, 전체 신경망이 잘 돌아가도록 이런 숫자들을 설정하는 과정은 매우 골치 아프다. 제로 패딩이나 다른 신경망 디자인 비법들을 사용하면 훨씬 수월하게 진행할 수 있다.

*실제 예제*. 이미지넷 대회에서 우승한 [Krizhevsky et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) 의 모델의 경우 [227x227x3] 크기의 이미지를 입력으로 받는다. 첫 번째 컨볼루션 레이어에서는 리셉티브 필드 $$F=11$$, stride $$S=4$$를 사용했고 제로 패딩은 사용하지 않았다 $$P=0$$. (227 - 11)/4 +1=55 이고 컨볼루션 레이어의 깊이는 $$K=96$$이므로 이 컨볼루션 레이어의 크기는 [55x55x96]이 된다. 각각의 55\*55\*96개 뉴런들은 입력 볼륨의 [11x11x3]개 뉴런들과 연결되어 있다. 그리고 각 깊이의 모든 96개 뉴런들은 입력 볼륨의 같은 [11x11x3] 영역에 서로 다른 가중치를 가지고 연결된다.

**파라미터 공유**. 파라미터 공유 기법은 컨볼루션 레이어의 파라미터 개수를 조절하기 위해 사용된다. 위의 실제 예제에서 보았듯, 첫 번째 컨볼루션 레이어에는 55\*55\*96 = 290,400 개의 뉴런이 있고 각각의 뉴런은 11\*11\*3 = 363개의 가중치와 1개의 바이어스를 가진다. 첫 번째 컨볼루션 레이어만 따져도 총 파라미터 개수는  290400*364=105,705,600개가 된다. 분명히 이 숫자는 너무 크다.

사실 적절한 가정을 통해 파라미터 개수를 크게 줄이는 것이 가능하다: (x,y)에서 어떤 patch feature가 유용하게 사용되었다면, 이 feature는 다른 위치 (x2,y2)에서도 유용하게 사용될 수 있다. 3차원 볼륨의 한 슬라이스 (깊이 차원으로 자른 2차원 슬라이스) 를 **depth slice**라고 하자 ([55x55x96] 사이즈의 볼륨은 각각 [55x55]의 크기를 가진 96개의 depth slice임). 앞으로는 각 depth slice 내의 뉴런들이 같은 가중치와 바이어스를 가지도록 제한할 것이다. 이런 파라미터 공유 기법을 사용하면, 예제의 첫 번째 컨볼루션 레이어는 (depth slice 당) 96개의 고유한 가중치를 가져서 총 96\*11\*11\*3 = 34,848개의 고유한 가중치, 또는 바이어스를 합쳐서 34,944개의 파라미터를 갖게 된다. 또는 각 depth slice에 존재하는 55*55개의 뉴런들은 모두 같은 파라미터를 사용하게 된다. 실제로는 backpropagation 과정에서 각 depth slice 내의 모든 뉴런들이 가중치에 대한 gradient를 계산하겠지만, 가중치 업데이트 할 때에는 이 gradient들을 합해 사용한다.

한 depth slice내의 모든 뉴런들이 같은 가중치 벡터를 갖기 때문에 컨볼루션 레이어의 forward pass는 입력 볼륨과 가중치 간의 **컨볼루션**으로 계산될 수 있다 (컨볼루션 레이어라는 이름이 붙은 이유).  그러므로 컨볼루션 레이어의 가중치는 **필터(filter)** 또는 **커널(kernel)**이라고 부른다. 컨볼루션의 결과물은 **액티베이션 맵(activation map, [55x55] 사이즈)** 이 되며 각 깊이에 해당하는 필터의 액티베이션 맵들을 쌓으면 최종 출력 볼륨 ([55x55x96] 사이즈) 가 된다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/cnn/weights.jpeg">
  <div class="figcaption">
    Krizhevsky et al. 에서 학습된 필터의 예. 96개의 필터 각각은 [11x11x3] 사이즈이며, 하나의 depth slice 내 55*55개 뉴런들이 이 필터들을 공유한다. 만약 이미지의 특정 위치에서 가로 엣지 (edge)를 검출하는 것이 중요했다면, 이미지의 다른 위치에서도 같은 특성이 중요할 수 있다 (이미지의 translationally-invariant한 특성 때문). 그러므로 55*55개 뉴런 각각에 대해 가로 엣지 검출 필터를 재학습 할 필요가 없다.
  </div>
</div>

가끔은 파라미터 sharing에 대한 가정이 부적절할 수도 있다. 특히 입력 이미지가 중심을 기준으로 찍힌 경우 (예를 들면 이미지 중앙에 얼굴이 있는 이미지), 이미지의 각 영역에 대해 완전히 다른 feature들이 학습되어야 할 수 있다. 눈과 관련된 feature나 머리카락과 관련된 feature 등은 서로 다른 영역에서 학습될 것이다. 이런 경우에는 파라미터 sharing 기법을 접어두고 대신 **Locally-Connected Layer**라는 레이어를 사용하는 것이 좋다.

**Numpy 예제.** 위에서 다룬 것들을 더 확실히 알아보기 위해 코드를 작성해보자. 입력 볼륨을 numpy 배열 `X`라고 하면:
- A *depth column* at position `(x,y)` would be the activations `X[x,y,:]`.
- `(x,y)`위치에서의 *depth column*은 액티베이션 `X[x,y,:]`이 된다.
- A *depth slice*, or equivalently an *activation map* at depth `d` would be the activations `X[:,:,d]`.
- depth `d`에서의 *depth slice*, 또는 *액티베이션 맵 (activation map)*은 `X[:,:,d]`가 된다.

*컨볼루션 레이어 예제*. 입력 볼륨 `X`의 모양이 `X.shape: (11,11,4)`이고 제로 패딩은 사용하지 않으며($$P = 0$$) 필터 크기는 $$F = 5$$, stride $$S = 2$$라고 하자. 출력 볼륨의 spatial 크기 (가로/세로)는 (11-5)/2 + 1 = 4가 된다. 출력 볼륨의 액티베이션 맵 (`V`라고 하자) 는 아래와 같은 것이다 (아래에는 일부 요소만 나타냄).

- `V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0`
- `V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0`
- `V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0`
- `V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0`

Numpy에서 `*`연산은 두 배열 간의 elementwise 곱셈이라는 것을 기억하자. 또한 `W0`는 가중치 벡터이고 `b0`은 바이어스라는 것도 기억하자. 여기에서 `W0`의 모양은 `W0.shape: (5,5,4)`라고 가정하자 (필터 사이즈는 5, depth는 4). 각 위치에서 일반 신경망에서와 같이 내적 연산을 수행하게 된다. 또한 파라미터 sharing 기법으로 같은 가중치, 바이어스가 사용되고 가로 차원에 대해 2 (stride)칸씩 옮겨가며 연산이 이뤄진다는 것을 볼 수 있다. 출력 볼륨의 두 번째 액티베이션 맵을 구성하는 방법은:

- `V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1`
- `V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1`
- `V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1`
- `V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1`
- `V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1` (example of going along y)
- `V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1` (or along both)

위 예제는 `V`의 두 번째 depth 차원 (인덱스 1)을 인덱싱하고 있다. 두 번째 액티베이션 맵을 계산하므로, 여기에서 사용된 가중치는 이전 예제와 달리 `W1`이다. 보통 액티베이션 맵이 구해진 뒤 ReLU와 같은 elementwise 연산이 가해지는 경우가 많은데, 위 예제에서는 다루지 않았다.

**요약**. To summarize, the Conv Layer:

- $$W_1 \times H_1 \times D_1$$ 크기의 볼륨을 입력받는다.
- 4개의 hyperparameter가 필요하다:
  - 필터 개수 $$K$$,
  - 필터의 가로/세로 Spatial 크기  $$F$$,
  - Stride $$S$$,
  - 제로 패딩 $$P$$.
- $$W_2 \times H_2 \times D_2$$ 크기의 출력 볼륨을 생성한다:
  - $$W_2 = (W_1 - F + 2P)/S + 1$$
  - $$H_2 = (H_1 - F + 2P)/S + 1$$ (i.e. 가로/세로는 같은 방식으로 계산됨)
  - $$D_2 = K$$
- 파라미터 sharing로 인해 필터 당 $$F \cdot F \cdot D_1$$개의 가중치를 가져서 총 $$(F \cdot F \cdot D_1) \cdot K$$개의 가중치와 $$K$$개의 바이어스를 갖게 된다.
- 출력 볼륨에서 $$d$$번째 depth slice ($$W_2 \times H_2$$ 크기)는 입력 볼륨에 $$d$$번째 필터를 stride $$S$$만큼 옮겨가며 컨볼루션 한 뒤 $$d$$번째 바이어스를 더한 결과이다.

흔한 Hyperparameter기본 세팅은 $$F = 3, S = 1, P = 1$$이다. 뒤에서 다룰 [ConvNet architectures](#architectures)에서 hyperparameter 세팅과 관련된 법칙이나 방식 등을 확인할 수 있다.

**컨볼루션 데모**. 아래는 컨볼루션 레이어 데모이다. 3차원 볼륨은 시각화하기 힘드므로 각 행마다 depth slice를 하나씩 배치했다. 각 볼륨은 입력 볼륨(파란색), 가중치 볼륨(빨간색), 출력 볼륨(녹색)으로 이뤄진다. 입력 볼륨의 크기는 $$W_1 = 5, H_1 = 5, D_1 = 3$$이고 컨볼루션 레이어의 파라미터들은 $$K = 2, F = 3, S = 2, P = 1$$이다. 즉, 2개의 $$3 \times 3$$크기의 필터가 각각 stride 2마다 적용된다. 그러므로 출력 볼륨의 spatial 크기 (가로/세로)는 (5 - 3 + 2)/2 + 1 = 3이다. 제로 패딩 $$P = 1$$ 이 적용되어 입력 볼륨의 가장자리가 모두 0으로 되어있다는 것을 확인할 수 있다. 아래의 영상에서 하이라이트 표시된 입력(파란색)과 필터(빨간색)이 elementwise로 곱해진 뒤 하나로 더해지고 bias가 더해지는걸 볼 수 있다.

<div class="fig figcenter fighighlight">
  <iframe src="{{site.baseurl}}/assets/conv-demo/index.html" width="100%" height="700px;" style="border:none;"></iframe>
  <div class="figcaption"></div>
</div>

**매트릭스 곱으로 구현**. 컨볼루션 연산은 필터와 이미지의 로컬한 영역간의 내적 연산을 한 것과 같다. 컨볼루션 레이어의 일반적인 구현 패턴은 이 점을 이용해 컨볼루션 레이어의 forward pass를 다음과 같이 하나의 큰 매트릭스 곱으로 계산된다:

1. 이미지의 각 로컬 영역을 열 벡터로 stretch 한다 (이런 연산을 보통 **im2col** 이라고 부름). 예를 들어, 만약 [227x227x3] 사이즈의 입력이 11x11x3 사이즈와 strie 4의 필터와 컨볼루션 한다면, 이미지에서 [11x11x3] 크기의 픽셀 블록을 가져와 11\*11\*3=363 크기의 열 벡터로 바꾸게 된다. 이 과정을 stride 4마다 하므로 가로, 세로에 대해 각각 (227-11)/4+1=55, 총 55\*55=3025 개 영역에 대해 반복하게 되고, 출력물인 `X_col`은 [363x3025]의 사이즈를 갖게 된다. 각각의 열 벡터는 리셉티브 필드를 1차원으로 stretch 한 것이고, 이 리셉티브 필드는 주위 리셉티브 필드들과 겹치므로 입력 볼륨의 여러 값들이 여러 출력 열벡터에 중복되어 나타날 수 있다.
2. 컨볼루션 레이어의 가중치는 비슷한 방식으로 행 벡터 형태로 stretch된다. 예를 들어 [11x11x3]사이즈의 총 96개 필터가 있다면, [96x363] 사이즈의 W_row가 만들어진다.
3. 이제 컨볼루션 연산은 하나의 큰 매트릭스 연산 `np.dot(W_row, X_col)`를 계산하는 것과 같다. 이 연산은 모든 필터와 모든 리셉티브 필터 영역들 사이의 내적 연산을 하는 것과 같다. 우리의 예에서는 각 영역에 대한 각각의 필터를 각각의 영역에 적용한 [96x3025] 사이즈의 출력물이 얻어진다.
4. 결과물은 [55x55x96] 차원으로 reshape 한다.

이 방식은 입력 볼륨의 여러 값들이 `X_col`에 여러 번 복사되기 때문에 메모리가 많이 사용된다는 단점이 있다. 그러나 매트릭스 연산과 관련된 많은 효율적 구현방식들을 사용할 수 있다는 장점도 있다 ([BLAS](http://www.netlib.org/blas/) API 가 하나의 예임). 뿐만 아니라 같은 *im2col* 아이디어는 풀링 연산에서 재활용 할 수도 있다 (뒤에서 다루게 된다).

**Backpropagation.** 컨볼루션 연산의 backward pass 역시 컨볼루션 연산이다 (가로/세로가 뒤집어진 필터를 사용한다는 차이점이 있음). 간단한 1차원 예제를 가지고 쉽게 확인해볼 수 있다.

<a name='pool'></a>
#### 풀링 레이어 (Pooling Layer)

CNN 구조 내에 컨볼루션 레이어들 중간중간에 주기적으로 풀링 레이어를 넣는 것이 일반적이다. 풀링 레이어가 하는 일은 네트워크의 파라미터의 개수나 연산량을 줄이기 위해 representation의 spatial한 사이즈를 줄이는 것이다. 이는 오버피팅을 조절하는 효과도 가지고 있다. 풀링 레이어는 MAX 연산을 각 depth slice에 대해 독립적으로 적용하여 spatial한 크기를 줄인다. 사이즈 2x2와 stride 2가 가장 많이 사용되는 풀링 레이어이다. 각 depth slice를 가로/세로축을 따라 1/2로 downsampling해 75%의 액티베이션은 버리게 된다. 이 경우 MAX 연산은 4개 숫자 중 최대값을 선택하게 된다 (같은 depth slice 내의 2x2 영역). Depth 차원은 변하지 않는다. 풀링 레이어의 특징들은 일반적으로 아래와 같다:

- $$W_1 \times H_1 \times D_1$$ 사이즈의 입력을 받는다
- 3가지 hyperparameter를 필요로 한다.
  - Spatial extent $$F$$
  - Stride $$S$$
- $$W_2 \times H_2 \times D_2$$ 사이즈의 볼륨을 만든다
  - $$W_2 = (W_1 - F)/S + 1$$
  - $$H_2 = (H_1 - F)/S + 1$$
  - $$D_2 = D_1$$
- 입력에 대해 항상 같은 연산을 하므로 파라미터는 따로 존재하지 않는다
- 풀링 레이어에는 보통 제로 패딩을 하지 않는다

일반적으로 실전에서는 두 종류의 max 풀링 레이어만 널리 쓰인다. 하나는 overlapping 풀링이라고도 불리는 $$F = 3, S = 2$$ 이고 하나는 더 자주 쓰이는 $$F = 2, S = 2$$ 이다. 큰 리셉티브 필드에 대해서 풀링을 하면 보통 너무 많은 정보를 버리게 된다.

**일반적인 풀링**. Max 풀링 뿐 아니라 *average 풀링*, *L2-norm 풀링* 등 다른 연산으로 풀링할 수도 있다. Average 풀링은 과거에 많이 쓰였으나 최근에는 Max 풀링이 더 좋은 성능을 보이며 점차 쓰이지 않고 있다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/cnn/pool.jpeg" width="36%">
  <img src="{{site.baseurl}}/assets/cnn/maxpool.jpeg" width="59%" style="border-left: 1px solid black;">
  <div class="figcaption">
    풀링 레이어는 입력 볼륨의 각 depth slice를 spatial하게 downsampling한다. <b>좌:</b> 이 예제에서는 입력 볼륨이 [224x224x64]이며 필터 크기 2, stride 2로 풀링해 [112x112x64] 크기의 출력 볼륨을 만든다. 볼륨의 depth는 그대로 유지된다는 것을 기억하자. <b>Right:</b> 가장 널리 쓰이는 <b>max 풀링</b>. 2x2의 4개 숫자에 대해 max를 취하게된다.
  </div>
</div>

**Backpropagation**. Backpropagation 챕터에서 max(x,y)의 backward pass는 그냥 forward pass에서 가장 큰 값을 가졌던 입력의 gradient를 보내는 것과 같다고 배운 것을 기억하자. 그러므로 forward pass 과정에서 보통 max 액티베이션의 위치를 저장해두었다가 backpropagation 때 사용한다.

**최근의 발전된 내용들**.

- [Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) 2x2보다 더 작은 필터들로 풀링하는 방식. 1x1, 1x2, 2x1, 2x2 크기의 필터들을 임의로 조합해 풀링한다. 매 forward pass마다 grid들이 랜덤하게 생성되고, 테스트 때에는 여러 grid들의 예측 점수들의 평균치를 사용하게 된다.
- [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806) 라는 논문은 컨볼루션 레이어만 반복하며 풀링 레이어를 사용하지 않는 방식을 제안한다. Representation의 크기를 줄이기 위해 가끔씩 큰 stride를 가진 컨볼루션 레이어를 사용한다.

풀링 레이어가 보통 representation의 크기를 심하게 줄이기 때문에 (이런 효과는 작은 데이터셋에서만 오버피팅 방지 효과 등으로 인해 도움이 됨), 최근 추세는 점점 풀링 레이어를 사용하지 않는 쪽으로 발전하고 있다.

<a name='norm'></a>
#### Normalization 레이어

실제 두뇌의 억제 메커니즘 구현 등을 위해 많은 종류의 normalization 레이어들이 제안되었다. 그러나 이런 레이어들이 실제로 주는 효과가 별로 없다는 것이 알려지면서 최근에는 거의 사용되지 않고 있다. Normalization에 대해 알고 싶다면 Alex Krizhevsky의 글을 읽어보기 바란다 [cuda-convnet library API](http://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)).

<a name='fc'></a>
#### Fully-connected 레이어

Fully connected 레이어 내의 뉴런들은 일반 신경망 챕터에서 보았듯이이전 레이어의 모든 액티베이션들과 연결되어 있다. 그러므로 Fully connected레이어의 액티베이션은 매트릭스 곱을 한 뒤 바이어스를 더해 구할 수 있다. 더 많은 정보를 위해 강의 노트의 "신경망" 섹션을 보기 바란다.

<a name='convert'></a>
#### FC 레이어를 CONV 레이어로 변환하기

FC 레이어와 CONV 레이어의 차이점은, CONV 레이어는 입력의 일부 영역에만 연결되어 있고, CONV 볼륨의 많은 뉴런들이 파라미터를 공유한다는 것 뿐이라는 것을 알아 둘 필요가 있다. 두 레이어 모두 내적 연산을 수행하므로 실제 함수 형태는 동일하다. 그러므로 FC 레이어를 CONV 레이어로 변환하는 것이 가능하다:

- 모든 CONV 레이어는 동일한 forward 함수를 수행하는 FC 레이어 짝이 있다. 이 경우의 가중치 매트릭스는 몇몇 블록을 제외하고 모두 0으로 이뤄지며 (local connectivity: 입력의 일부 영역에만 연결된 특성), 이 블록들 중 여러개는 같은 값을 지니게 된다 (파라미터 공유).

- 반대로, 모든 FC 레이어는 CONV 레이어로 변환될 수 있다. 예를 들어, $$7 \times 7 \times 512$$ 크기의 입력을 받고 $$K= 4906$$ 인 FC 레이어는 $$F = 7, P = 0, S = 1, K = 4096$$인 CONV 레이어로 표현 가능하다. 바꿔 말하면, 필터의 크기를 입력 볼륨의 크기와 동일하게 만들고  $$1 \times 1 \times 4906$$ 크기의 아웃풋을 출력할 수 있다. 각 depth에 대해 하나의 값만 구해지므로 (필터의 가로/세로가 입력 볼륨의 가로/세로와 같으므로) FC 레이어와 같은 결과를 얻게 된다.

**FC->CONV 변환**. 이 두 변환 중, FC 레이어를 CONV 레이어로의 변환은 매우 실전에서 매우 유용하다. 224x224x3의 이미지를 입력으로 받고 일련의 CONV레이어와 POOL 레이어를 이용해 7x7x512의 액티베이션을 만드는 컨볼루션넷 아키텍쳐를 생각해 보자 (뒤에서 살펴 볼 *AlexNet* 아키텍쳐에서는 입력의 spatial(가로/세로) 크기를 반으로 줄이는 풀링 레이어 5개를 사용해 7x7x512의 액티베이션을 만든다. 224/2/2/2/2/2 = 7이기 때문이다). AlexNet은 여기에 4096의 크기를 갖는 FC 레이어 2개와 클래스 스코어를 계산하는 1000개 뉴런으로 이뤄진 마지막 FC 레이어를 사용한다. 이 마지막 3개의 FC 레이어를 CONV 레이어로 변환하는 방법을 아래에서 배우게 된다:

- [7x7x512]의 입력 볼륨을 받는 첫 번째 FC 레이어를 $$F = 7$$의 필터 크기를 갖는 CONV 레이어로 바꾼다. 이 때 출력 볼륨의 크기는 [1x1x4096] 이 된다.
- 두 번째 FC 레이어를 $$F = 1$$ 필터 사이즈의 CONV 레이어로 바꾼다. 이 때 출력 볼륨의 크기는 [1x1s4096]이 된다.
- 같은 방식으로 마지막 FC 레이어를 $$F = 1$$의 CONV 레이어를 바꾼다. 출력 볼륨의 크기는 [1x1x1000]이 된다.

각각의 변환은 일반적으로 FC 레이어의 가중치 $$W$$를 CONV 레이어의 필터로 변환하는 과정을 수반한다. 이런 변환을 하고 나면, 큰 이미지 (가로/세로가 224보다 큰 이미지)를 단 한번의 forward pass만으로 마치 이미지를 "슬라이딩"하면서 여러 영역을 읽은 것과 같은 효과를 준다.

예를 들어,224x224 크기의 이미지를 입력으로 받으면 [7x7x512]의 볼륨을 출력하는 이 아키텍쳐에, ( 224/7 = 32배 줄어듦 ) 된 아키텍쳐에 384x384 크기의 이미지를 넣으면 [12x12x512] 크기의 볼륨을 출력하게 된다 (384/32 = 12 이므로). 이후 FC에서 CONV로 변환한 3개의 CONV 레이어를 거치면 [6x6x1000] 크기의 최종 볼륨을 얻게 된다 ( (12 - 7)/1 +1 =6 이므로). [1x1x1000]크기를 지닌 하나의 클래스 점수 벡터 대신 384x384 이미지로부터 6x6개의 클래스 점수 배열을 구했다는 것이 중요하다.

For example, if 224x224 image gives a volume of size [7x7x512] - i.e. a reduction by 32, then forwarding an image of size 384x384 through the converted architecture would give the equivalent volume in size [12x12x512], since 384/32 = 12. Following through with the next 3 CONV layers that we just converted from FC layers would now give the final volume of size [6x6x1000], since (12 - 7)/1 + 1 = 6. Note that instead of a single vector of class scores of size [1x1x1000], we're now getting and entire 6x6 array of class scores across the 384x384 image.

> 위의 내용은 384x384 크기의 이미지를 32의 stride 간격으로 224x224 크기로 잘라 각각을 원본 ConvNet (뒷쪽 3개 레이어가 FC인)에 적용한 것과 같은 결과를 보여준다.

당연히 (CONV레이어만으로) 변환된 ConvNet을 이용해 한 번에 이미지를 처리하는 것이 원본 ConvNet으로 36개 위치에 대해 반복적으로 처리하는 것 보다 훨씬 효율적이다. 36번의 처리 과정에서 같은 계산이 중복되기 때문이다. 이런 기법은 실전에서 성능 향상을 위해 종종 사용된다. 예를 들어 이미지를 크게 리사이즈 한 뒤 변환된 ConvNet을 이용해 여러 위치에 대한 클래스 점수를 구한 다음 그 점수들의 평균을 취하는 기법 등이 있다.

마지막으로 32 픽셀보다 적은 stride 간격으로 ConvNet을 적용하고 싶다면 어떡해야 할까? 포워드 패스 (forward pass)를 여러 번 적용하면 가능하다. 예를 들어 16의 stride 간격으로 처리를 하고 싶다면 변환된 ConvNet에 이미지를 2번 적용한 뒤 합치는 방식을 사용하면 된다: 먼저 원본 이미지를 처리한 뒤 원본 이미지를 가로/세로 16 픽셀만큼 쉬프트 시킨 뒤 한번 더 처리하면 된다.

- Caffe를 이용해 ConvNet 변환을 수행하는 실제 IPython Notebook 예제 [Net Surgery](https://github.com/BVLC/caffe/blob/master/examples/net_surgery.ipynb)

<a name='architectures'></a>
### ConvNet 구조

위에서 컨볼루셔널 신경망은 일반적으로 CONV, POOL (별다른 언급이 없다면 Max Pool이라고 가정), FC 레이어로 이뤄져 있다는 것을 배웠다. 각 원소에 비선형 특징을 가해주는 RELU 액티베이션 함수도 명시적으로 레이어로 취급하겠다. 이 섹션에서는 어떤 방식으로 이 레이어들이 쌓아져 전체 ConvNet이 이뤄지는지 알아보겠다.

<a name='layerpat'></a>
#### 레이어 패턴
가장 흔한 ConvNet 구조는 몇 개의 CONV-RELU 레이어를 쌓은 뒤 POOL 레이어를 추가한 형태가 여러 번 반복되며 이미지 볼륨의 spatial (가로/세로) 크기를 줄이는 것이다. 이런 방식으로 적절히 쌓은 뒤 FC 레이어들을 쌓아준다. 마지막 FC 레이어는 클래스 점수와 같은 출력을 만들어낸다. 다시 말해서, 일반적인 ConvNet 구조는 다음 패턴을 따른다:
`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

`*`는 반복을 의미하며 `POOL?` 은 선택적으로 POOL 레이어를 사용한다는 의미이다. 또한 `N >= 0` (보통 `N <= 3`), `M >= 0`, `K >= 0` (보통 `K < 3`)이다. 예를 들어, 보통의 ConvNet 구조에서 아래와 같은 패턴들을 흔히 발견할 수 있다:

- `INPUT -> FC`, 선형 분류기이다. 이 때 `N = M = K = 0`.
- `INPUT -> CONV -> RELU -> FC`
- `INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC`. 이 경우는 POOL 레이어 하나 당 하나의 CONV 레이어가 존재한다.
- `INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC` 이 경우는 각각의 POOL 레이어를 거치기 전에 여러 개의 CONV 레이어를 거치게 된다. 크고 깊은 신경망에서는 이런 구조가 적합하다. 여러 층으로 쌓인 CONV 레이어는 pooling 연산으로 인해 많은 정보가 파괴되기 전에 복잡한 feature들을 추출할 수 있게 해주기 때문이다.

*큰 리셉티브 필드를 가지는 CONV 레이어 하나 대신 여러개의 작은 필터를 가진 CONV 레이어를 쌓는 것이 좋다*. 3x3 크기의 CONV 레이어 3개를 쌓는다고 생각해보자 (물론 각 레이어 사이에는 비선형 함수를 넣어준다). 이 경우 첫 번째 CONV 레이어의 각 뉴런은 입력 볼륨의 3x3 영역을 보게 된다. 두 번째 CONV 레이어의 각 뉴런은 첫 번째 CONV 레이어의 3x3 영역을 보게 되어 결론적으로 입력 볼륨의 5x5 영역을 보게 되는 효과가 있다. 비슷하게, 세 번째 CONV 레이어의 각 뉴런은 두 번째 CONV 레이어의 3x3 영역을 보게 되어 입력 볼륨의 7x7 영역을 보는 것과 같아진다. 이런 방식으로 3개의 3x3 CONV 레이어를 사용하는 대신 7x7의 리셉티브 필드를 가지는 CONV 레이어 하나를 사용한다고 생각해 보자. 이 경우에도 각 뉴런은 입력 볼륨의 7x7 영역을 리셉티브 필드로 갖게 되지만 몇 가지 단점이 존재한다. 먼저, CONV 레이어 3개를 쌓은 경우에는 중간 중간 비선형 함수의 영향으로 표현력 높은 feature를 만드는 반면, 하나의 (7x7) CONV 레이어만 갖는 경우 각 뉴런은 입력에 대해 선형 함수를 적용하게 된다. 두 번째로, 모든 볼륨이 $$C$$ 개의 채널(또는 깊이)을 갖는다고 가정한다면, 7x7 CONV 레이어의 경우 $$C \times (7 \times 7 \times C)=49 C^2$$개의 파라미터를 갖게 된다. 반면 3개의 3x3 CONV 레이어의 경우는 $$3 \times (C \times (3 \times 3 \times)) = 27 C^2$$개의 파라미터만 갖게 된다. 직관적으로, 하나의 큰 필터를 갖는 CONV 레이어보다, 작은 필터를 갖는 여러 개의 CONV 레이어를 쌓는 것이 더 적은 파라미터만 사용하면서도 입력으로부터 더 좋은 feature를 추출하게 해준다. 단점이 있다면, backpropagation을 할 때 CONV 레이어의 중간 결과들을 저장하기 위해 더 많은 메모리 공간을 잡고 있어야 한다는 것이다. 

<a name='layersizepat'></a>
#### Layer Sizing Patterns

Until now we've omitted mentions of common hyperparameters used in each of the layers in a ConvNet. We will first state the common rules of thumb for sizing the architectures and then follow the rules with a discussion of the notation:

The **input layer** (that contains the image) should be divisible by 2 many times. Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512.

The **conv layers** should be using small filters (e.g. 3x3 or at most 5x5), using a stride of $$S = 1$$, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. That is, when $$F = 3$$, then using $$P = 1$$ will retain the original size of the input. When $$F = 5$$, $$P = 2$$. For a general $$F$$, it can be seen that $$P = (F - 1) / 2$$ preserves the input size. If you must use bigger filter sizes (such as 7x7 or so), it is only common to see this on the very first conv layer that is looking at the input image.

The **pool layers** are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. $$F = 2$$), and with a stride of 2 (i.e. $$S = 2$$). Note that this discards exactly 75% of the activations in an input volume (due to downsampling by 2 in both width and height). Another sligthly less common setting is to use 3x3 receptive fields with a stride of 2, but this makes. It is very uncommon to see receptive field sizes for max pooling that are larger than 3 because the pooling is then too lossy and agressive. This usually leads to worse performance.

*Reducing sizing headaches.* The scheme presented above is pleasing because all the CONV layers preserve the spatial size of their input, while the POOL layers alone are in charge of down-sampling the volumes spatially. In an alternative scheme where we use strides greater than 1 or don't zero-pad the input in CONV layers, we would have to very carefully keep track of the input volumes throughout the CNN architecture and make sure that all strides and filters "work out", and that the ConvNet architecture is nicely and symmetrically wired.

*Why use stride of 1 in CONV?* Smaller strides work better in practice. Additionally, as already mentioned stride 1 allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise.

*Why use padding?* In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be "washed away" too quickly.

*Compromising based on memory constraints.* In some cases (especially early in the ConvNet architectures), the amount of memory can build up very quickly with the rules of thumb presented above. For example, filtering a 224x224x3 image with three 3x3 CONV layers with 64 filters each and padding 1 would create three activation volumes of size [224x224x64]. This amounts to a total of about 10 million activations, or 72MB of memory (per image, for both activations and gradients). Since GPUs are often bottlenecked by memory, it may be necessary to compromise. In practice, people prefer to make the compromise at only the first CONV layer of the network. For example, one compromise might be to use a first CONV layer with filter sizes of 7x7 and stride of 2 (as seen in a ZF net). As another example, an AlexNet uses filer sizes of 11x11 and stride of 4.

<a name='case'></a>
#### Case studies

There are several architectures in the field of Convolutional Networks that have a name. The most common are:

- **LeNet**. The first successful applications of Convolutional Networks were developed by Yann LeCun in 1990's. Of these, the best known is the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture that was used to read zip codes, digits, etc.
- **AlexNet**. The first work that popularized Convolutional Networks in Computer Vision was the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), developed by Alex Krizhevsky, Ilya Sutskever and Geoff Hinton. The AlexNet was submitted to the [ImageNet ILSVRC challenge](http://www.image-net.org/challenges/LSVRC/2014/) in 2012 and significantly outperformed the second runner-up (top 5 error of 16% compared to runner-up with 26% error). The Network had a similar architecture basic as LeNet, but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer immediately followed by a POOL layer).
- **ZF Net**. The ILSVRC 2013 winner was a Convolutional Network from Matthew Zeiler and Rob Fergus. It became known as the [ZFNet](http://arxiv.org/abs/1311.2901) (short for Zeiler & Fergus Net). It was an improvement on AlexNet by tweaking the architecture hyperparameters, in particular by expanding the size of the middle convolutional layers.
- **GoogLeNet**. The ILSVRC 2014 winner was a Convolutional Network from [Szegedy et al.](http://arxiv.org/abs/1409.4842) from Google. Its main contribution was the development of an *Inception Module* that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M). Additionally, this paper uses Average Pooling instead of Fully Connected layers at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much.
- **VGGNet**. The runner-up in ILSVRC 2014 was the network from Karen Simonyan and Andrew Zisserman that became known as the [VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). Its main contribution was in showing that the depth of the network is a critical component for good performance. Their final best network contains 16 CONV/FC layers and, appealingly, features an extremely homogeneous architecture that only performs 3x3 convolutions and 2x2 pooling from the beginning to the end. It was later found that despite its slightly weaker classification performance, the VGG ConvNet features outperform those of GoogLeNet in multiple transfer learning tasks. Hence, the VGG network is currently the most preferred choice in the community when extracting CNN features from images. In particular, their [pretrained model](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) is available for plug and play use in Caffe. A downside of the VGGNet is that it is more expensive to evaluate and uses a lot more memory and parameters (140M).
- **ResNet**. [Residual Network](http://arxiv.org/abs/1512.03385) developed by Kaiming He et al. was the winner of ILSVRC 2015. It features an interesting architecture with special *skip connections* and features heavy use of batch normalization. The architecture is also missing fully connected layers at the end of the network. The reader is also referred to Kaiming's presentation ([video](https://www.youtube.com/watch?v=1PGLj-uKT1w), [slides](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)), and some [recent experiments](https://github.com/gcr/torch-residual-networks) that reproduce these networks in Torch.

**VGGNet in detail**.
Lets break down the [VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) in more detail. The whole VGGNet is composed of CONV layers that perform 3x3 convolutions with stride 1 and pad 1, and of POOL layers that perform 2x2 max pooling with stride 2 (and no padding). We can write out the size of the representation at each step of the processing and keep track of both the representation size and the total number of weights:

~~~
INPUT: [224x224x3]        memory:  224*224*3=150K   weights: 0
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*3)*64 = 1,728
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*64)*64 = 36,864
POOL2: [112x112x64]  memory:  112*112*64=800K   weights: 0
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*128)*128 = 147,456
POOL2: [56x56x128]  memory:  56*56*128=400K   weights: 0
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2: [28x28x256]  memory:  28*28*256=200K   weights: 0
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2: [14x14x512]  memory:  14*14*512=100K   weights: 0
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2: [7x7x512]  memory:  7*7*512=25K  weights: 0
FC: [1x1x4096]  memory:  4096  weights: 7*7*512*4096 = 102,760,448
FC: [1x1x4096]  memory:  4096  weights: 4096*4096 = 16,777,216
FC: [1x1x1000]  memory:  1000 weights: 4096*1000 = 4,096,000

TOTAL memory: 24M * 4 bytes ~= 93MB / image (only forward! ~*2 for bwd)
TOTAL params: 138M parameters
~~~

As is common with Convolutional Networks, notice that most of the memory is used in the early CONV layers, and that most of the parameters are in the last FC layers. In this particular case, the first FC layer contains 100M weights, out of a total of 140M.


<a name='comp'></a>

#### Computational Considerations

The largest bottleneck to be aware of when constructing ConvNet architectures is the memory bottleneck. Many modern GPUs have a limit of 3/4/6GB memory, with the best GPUs having about 12GB of memory. There are three major sources of memory to keep track of:

- From the intermediate volume sizes: These are the raw number of **activations** at every layer of the ConvNet, and also their gradients (of equal size). Usually, most of the activations are on the earlier layers of a ConvNet (i.e. first Conv Layers). These are kept around because they are needed for backpropagation, but a clever implementation that runs a ConvNet only at test time could in principle reduce this by a huge amount, by only storing the current activations at any layer and discarding the previous activations on layers below.
- From the parameter sizes: These are the numbers that hold the network **parameters**, their gradients during backpropagation, and commonly also a step cache if the optimization is using momentum, Adagrad, or RMSProp. Therefore, the memory to store the parameter vector alone must usually be multiplied by a factor of at least 3 or so.
- Every ConvNet implementation has to maintain **miscellaneous** memory, such as the image data batches, perhaps their augmented versions, etc.

Once you have a rough estimate of the total number of values (for activations, gradients, and misc), the number should be converted to size in GB. Take the number of values, multiply by 4 to get the raw number of bytes (since every floating point is 4 bytes, or maybe by 8 for double precision), and then divide by 1024 multiple times to get the amount of memory in KB, MB, and finally GB. If your network doesn't fit, a common heuristic to "make it fit" is to decrease the batch size, since most of the memory is usually consumed by the activations.

### Visualizing and Understanding Convolutional Networks

In the [next section](../understanding-cnn/) of these notes we look at visualizing and understanding Convolutional Neural Networks.

<a name='add'></a>

### Additional Resources

Additional resources related to implementation:

- [DeepLearning.net tutorial](http://deeplearning.net/tutorial/lenet.html) walks through an implementation of a ConvNet in Theano
- [cuda-convnet2](https://code.google.com/p/cuda-convnet2/) by Alex Krizhevsky is a ConvNet implementation that supports multiple GPUs
- [ConvNetJS CIFAR-10 demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) allows you to play with ConvNet architectures and see the results and computations in real time, in the browser.
- [Caffe](http://caffe.berkeleyvision.org/), one of the most popular ConvNet libraries.
- [Example Torch 7 ConvNet](https://github.com/nagadomi/kaggle-cifar10-torch7) that achieves 7% error on CIFAR-10 with a single model
- [Ben Graham's Sparse ConvNet](https://www.kaggle.com/c/cifar-10/forums/t/10493/train-you-very-own-deep-convolutional-network/56310) package, which Ben Graham used to great success to achieve less than 4% error on CIFAR-10.

---
<p style="text-align:right"><b>
번역: 김택수 <a href="https://github.com/jazzsaxmafia" style="color:black">(jazzsaxmafia)</a>
</b></p>
