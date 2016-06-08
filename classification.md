---
layout: page
mathjax: true
permalink: /classification/
---

본 강의노트는 컴퓨터비전 외의 분야를 공부하던 사람들에게 Image Classification(이미지 분류) 문제와,  data-driven approach(데이터 기반 방법론)을 소개한다. 목차는 다음과 같다.

- [Image Classification(이미지 분류), data-driven approach(데이터 기반 방법론), pipeline(파이프라인)](#intro)
- [Nearest Neighbor 분류기](#nn)
  - [k-Nearest Neighbor 알고리즘](#knn)
- [Validation sets, Cross-validation, hyperparameter 튜닝](#val)
- [Nearest Neighbor의 장단점](#procon)
- [요약](#summary)
- [요약: 실제 문제에 kNN 적용하기](#summaryapply)
- [읽을 자료](#reading)

<a name='intro'></a>

## Image Classification(이미지 분류)

**동기**. 이 섹션에서는 이미지 분류 문제에 대해 다룰 것이다. 이미지 분류 문제란, 입력 이미지를 미리 정해진 카테고리 중 하나인 라벨로 분류하는 문제다. 문제 정의는 매우 간단하지만 다양한 활용 가능성이 있는 컴퓨터 비전 분야의 핵심적인 문제 중의 하나이다. 강의의 나중 파트에서도 살펴보겠지만, 이미지 분류와 멀어보이는 다른 컴퓨터 비전 분야의 여러 문제들 (물체 검출, 영상 분할 등)이 이미지 분류 문제를 푸는 것으로 인해 해결될 수 있다.

**예시**. 예를 들어, 아래 그림의 이미지 분류 모델은 하나의 이미지와 4개의 분류가능한 라벨 *{cat, dog, hat, mug}*  이 있다. 그림에서 보다시피, 컴퓨터에서 이미지는 3차원 배열로 표현된다. 이 예시에서 고양이 이미지는 가로 248픽셀(모니터의 화면을 구성하는 최소 단위, 역자 주), 세로 400픽셀로 구성되어 있고 Red, Green, Blue(RGB) 3개의 색상 채널이 있다. 따라서 이 이미지는 248 x 400 x 3개(총 297,500개)의 픽셀로 구성되어 있다. 각 픽셀의 값은 0~255 범위의 정수값이다. 이미지 분류 문제는 이 수많은 값들을 *"cat"* 이라는 하나의 라벨로 변경하는 것이다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/classify.png">
  <div class="figcaption">이미지 분류는 이미지가 주어졌을 때 그에 대한 라벨(각 라벨에 대한 신뢰도를 표시하는 분류)을 예측하는 일이다. 이미지는 0~255 정수 범위의 값을 가지는 Width(너비) x Height(높이) x 3의 크기의 3차원 배열이다. 3은 Red, Green, Blue로 구성된 3개의 채널을 의미한다.</div>
</div>

**문제**. 이미지를 분류하는 일(예를들어 *"cat"*)이 사람에게는 대수롭지 않겠지만, 컴퓨터 비전의 관점에서 생각해보면 해결해야 하는 문제들이 있다. 아래에 서술된 해결해야 하는 문제들처럼, 이미지는 3차원 배열의 값으로 나타내는 것을 염두해두어야 한다.

- **Viewpoint variation(시점 변화)**. 객체의 단일 인스턴스는 카메라에 의해 시점이 달라질 수 있다.
- **Scale variation(크기 변화)**. 비주얼 클래스는 대부분 그것들의 크기의 변화를 나타낸다(이미지의 크기뿐만 아니라 실제 세계에서의 크기까지 포함함).
- **Deformation(변형)**. 많은 객체들은 고정된 형태가 없고, 극단적인 형태로 변형될 수 있다.
- **Occlusion(폐색)**. 객체들은 전체가 보이지 않을 수 있다. 때로는 물체의 매우 적은 부분(매우 적은 픽셀)만이 보인다.
- **Illumination conditions(조명 상태)**. 조명의 영향으로 픽셀 값이 변형된다.
- **Background clutter(배경 분규)**. 객체가 주변 환경에 섞여(*blend*) 알아보기 힘들게 된다.
- **Intra-class variation(내부클래스의 다양성)**. 분류해야할 클래스는 범위가 큰 것들이 많다. 예를 들어 *의자* 의 경우, 매우 다양한 형태의 객체가 있다.

좋은 이미지 분류기는 각 클래스간의 감도를 유지하면서 동시에 이런 다양한 문제들에 대해 변함 없이 분류할 수 있는 성능을 유지해야 한다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/challenges.jpeg">
  <div class="figcaption"></div>
</div>

**Data-driven approach(데이터 기반 방법론)**. 어떻게 하면 이미지를 각각의 카테고리로 분류하는 알고리즘을 작성할 수 있을까? 숫자를 정렬하는 알고리즘 작성과는 달리 고양이를 분별하는 알고리즘을 작성하는 것은 어렵다.

 그러므로, 코드를 통해 직접적으로 모든 것을 카테고리로 분류하기 보다는 좀 더 쉬운 방법을 사용할 것이다. 먼저 컴퓨터에게 각 클래스에 대해 많은 예제를 주고 나서 이 예제들을 보고 시각적으로 학습할 수 있는 학습 알고리즘을 개발한다.
 이런 방법을 *data-driven approach(데이터 기반  방법론)* 이라고 한다. 이 방법은 라벨화가 된 이미지들 *training dataset(학습 데이터셋)* 이 처음 학습을 위해 필요하다. 아래 그림은 이런 데이터셋의 예이다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/trainset.jpg">
  <div class="figcaption">4개의 카테고리에 대한 학습 데이터셋에 대한 예. 학습과정에서 천여 개의 카테고리와 각 카테고리당 수십만 개의 이미지가 있을 수 있다.</div>
</div>

**The image classification pipeline(이미지 분류 파이프라인)**. 이미지 분류 문제란, 이미지를 픽셀들의 배열로 표현하고 각 이미지에 라벨을 하나씩 할당하는 문제라는 것을 이제까지 살펴보았다. 완전한 파이프라인은 아래와 같이 공식화할 수 있다:

- **Input(입력):** 입력은 *N* 개의 이미지로 구성되어 있고, *K* 개의 별개의 클래스로 라벨화 되어 있다. 이 데이터를 *training set* 으로 사용한다.
- **Learning(학습):** 학습에서 할 일은 트레이닝 셋을 이용해 각각의 클래스를 학습하는 것이다. 이 과정을 *training a classifier* 혹은 *learning a model* 이란 용어를 사용해 표현할 수 있다.
- **Evaluation(평가):** 마지막으로 새로운 이미지에 대해 어떤 라벨로 분류되는지 예측해봄으로써 분류기의 성능을 평가한다. 새로운 이미지의 라벨과 분류기를 통해 예측된 라벨을 비교할 것이다. 직관적으로, 많은 예상치들이 실제 답과 일치하기를 기대하는 것이고, 이 것을 우리는 *ground truth(실측 자료)* 라고 한다.

<a name='nn'></a>

## Nearest Neighbor Classifier(최근접 이웃 분류기)

첫번째 방법으로써, **Nearest Neighbor Classifier** 라 불리는 분류기를 개발할 것이다. 이 분류기는 컨볼루션 신경망 방법과는 아무 상관이 없고 실제 문제를 풀 때 자주 사용되지는 않지만, 이미지 분류 문제에 대한 기본적인 접근 방법을 알 수 있도록 한다.

**이미지 분류 데이터셋의 예: CIFAR-10.** 간단하면서 유명한 이미지 분류 데이터셋 중의 하나는 <a href="http://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10 dataset</a> 이다. 이 데이터셋은 60,000개의 작은 이미지로 구성되어 있고, 각 이미지는 32x32 픽셀 크기이다. 각 이미지는 10개의 클래스중 하나로 라벨링되어 있다(Ex. *"airplane, automobile, bird, etc"*). 이 60,000개의 이미지 중에 50,000개는 학습 데이터셋 (트레이닝 셋), 10,000개는 테스트 (데이터)셋으로 분류된다. 아래의 그림에서 각 10개의 클래스에 대해 임의로 선정한 10개의 이미지들의 예를 볼 수 있다:

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/nn.jpg">
  <div class="figcaption">좌: <a href="http://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10 dataset</a> 의 각 클래스 예. 우: 첫번째 열은 테스트 셋이고 나머지 열은 이 테스트 셋에 대해서 트레이닝 셋에 있는 이미지 중 픽셀값 차에 따른 상위 10개의 최근접 이웃 이미지이다.</div>
</div>

50,000개의 CIFAR-10 트레이닝 셋(하나의 라벨 당 5,000개의 이미지)이 주어진 상태에서 나머지 10,000개의 이미지에 대해 라벨화 하는 것을 가정해보자. 최근접 이웃 분류기는 테스트 이미지를 취해 모든 학습 이미지와 비교를 하고 라벨 값을 예상할 것이다. 상단 이미지의 우측과 같이 10개의 테스트 이미지에 대한 결과를 확인해보면, 10개의 이미지 중 3개만이 같은 클래스로 검색된 반면, 7개의 이미지는 같은 클래스로 분류되지 않았다. 예를 들어, 8번째 행의 말 학습 이미지에 대한 첫번째 최근접 이웃 이미지는 붉은색의 차이다. 짐작컨데 이 경우는 검은색 배경의 영향이 큰 듯 하다. 결과적으로, 이 말 이미지는 차로 잘못 분류될 것이다.

두개의 이미지(이 경우에는 32 x 32 x 3 크기의 두 블록)를 비교하는 정확한 방법을 아직 명시하지 않았다는 점을 눈치챘을 것이다. 가장 간단한 방법 중 하나는 이미지를 각각의 픽셀값으로 비교하고, 그 차이를 모두 더하는 것이다. 다시 말해서 두 개의 이미지가 주어지고 그 것들을 $$ I_1, I_2 $$ 벡터로 나타냈을 때, 벡터 간의 **L1 distance(L1 거리)** 를 계산하는 것이 한 가지 방법이다:

$$
d_1 (I_1, I_2) = \sum_{p} \left| I^p_1 - I^p_2 \right|
$$

결과는 모든 픽셀값 차이의 합이다. 아래에 그 과정을 시각화 하였다:

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/nneg.jpeg">
  <div class="figcaption">두 개의 이미지를 (각각의 색 채널마다의) L1 거리를 이용해서 비교할 때, 각 픽셀마다의 차이를 사용하는 예시. 두 이미지 벡터(행렬)의 각 성분마다 차를 계산하고, 그 차를 전부 더해서 하나의 숫자를 얻는다. 두 이미지가 똑같을 경우에는 결과가 0일 것이고, 두 이미지가 매우 다르다면 결과값이 클 것이다.</div>
</div>

다음으로, 분류기를 실제로 코드 상에서 어떻게 구현하는지 살펴보자. 첫 번째로 CIFAR-10 데이터를 메모리로 불러와 4개의 배열에 저장한다. 각각은 학습(트레이닝) 데이터와 그 라벨, 테스트 데이터와 그 라벨이다. 아래 코드에 `Xtr`(크기 50,000 x 32 x 32 x 3)은 트레이닝 셋의 모든 이미지를 저장하고 1차원 배열인 `Ytr`(길이 50,000)은 트레이닝 데이터의 라벨(0부터 9까지)을 저장한다.

~~~python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # 제공되는 함수
# 모든 이미지가 1차원 배열로 저장된다.
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows는 50000 x 3072 크기의 배열.
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows는 10000 x 3072 크기의 배열.
~~~

이제 모든 이미지를 배열의 각 행들로 얻었다. 아래에는 분류기를 어떻게 학습시키고 평가하는지에 대한 코드이다:

~~~python
nn = NearestNeighbor() # Nearest Neighbor 분류기 클래스 생성
nn.train(Xtr_rows, Ytr) # 학습 이미지/라벨을 활용하여 분류기 학습
Yte_predict = nn.predict(Xte_rows) # 테스트 이미지들에 대해 라벨 예측
# 그리고 분류 성능을 프린트한다
# 정확도는 이미지가 올바르게 예측된 비율로 계산된다 (라벨이 같을 비율)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
~~~

일반적으로 평가 기준으로서 **accuracy(정확도)** 를 사용한다. 정확도는 예측값이 실제와 얼마나 일치하는지 그 비율을 측정한다. 앞으로 만들어볼 모든 분류기는 공통적인 API를 갖게 될 것이다: 데이터(X)와 데이터가 실제로 속하는 라벨(y)을 입력으로 받는 `train(X,y)` 형태의 함수가 있다는 점이다. 내부적으로, 이 함수는 라벨들을 활용하여 어떤 모델을 만들어야 하고, 그 값들이 데이터로부터 어떻게 예측될 수 있는지를 알아야 한다. 그 이후에는 새로운 데이터로 부터 라벨을 예측하는 `predict(X)` 형태의 함수가 있다. 물론, 아직까지는 실제 분류기 자체가 빠져있다. 다음은 앞의 형식을 만족하는 L1 거리를 이용한 간단한 최근접 이웃 분류기의 구현이다:

~~~python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # nearest neighbor 분류기는 단순히 모든 학습 데이터를 기억해둔다.
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # 출력 type과 입력 type이 갖게 되도록 확인해준다.
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # i번째 테스트 이미지와 가장 가까운 학습 이미지를
      # L1 거리(절대값 차의 총합)를 이용하여 찾는다.
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # 가장 작은 distance를 갖는 인덱스를 찾는다.
      Ypred[i] = self.ytr[min_index] # 가장 가까운 이웃의 라벨로 예측

    return Ypred
~~~

이 코드를 실행해보면 이 분류기는 CIFAR-10에 대해 정확도가 **38.6%** 밖에 되지 않는다는 것을 확인할 수 있다. 임의로 답을 결정하는 것(10개의 클래스가 있으므로 10%의 정확도)보다는 낫지만, 사람의 정확도([약 94%](http://karpathy.github.io/2011/04/27/manually-classifying-cifar10/))나 최신 컨볼루션 신경망의 성능(약 95%)에는 훨씬 미치지 못한다(최근 Kaggle 대회 [순위표](http://www.kaggle.com/c/cifar-10/leaderboard) 참고).

**거리(distance) 선택**
벡터간의 거리를 계산하는 방법은 L1 거리 외에도 매우 많다. 또 다른 일반적인 선택으로, 기하학적으로 두 벡터간의 유클리디안 거리를 계산하는 것으로 해석할 수 있는 **L2 distance(L2 거리)** 의 사용을 고려해볼 수 있다. 이 거리의 계산 방식은 다음과 같다:

$$
d_2 (I_1, I_2) = \sqrt{\sum_{p} \left( I^p_1 - I^p_2 \right)^2}
$$

즉, 이전처럼 각 픽셀간의 차를 구하지만 각각에 제곱을 취하고, 전부 더한 다음에 최종적으로 제곱근을 취한다. NumPy를 사용한다면 위 코드를 사용하여 거리를 계산하는 아래의 코드 부분 딱 한 줄만 바꾸면 된다.

~~~python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
~~~

위 코드에서는 `np.sqrt` 함수를 호출하는 것을 그대로 남겨두었지만, 제곱근 함수는 단조 함수이기 때문에 실제 nearest neighbor 응용에서 제곱근은 빼도 결과에 상관이 없다. 즉, 계산되는 거리들의 크기에는 차이가 생기겠지만 그 순서는 동일하기 때문에, 제곱근 함수를 포함할 때와 포함하지 않을 때의 nearest neighbor(최근접 이웃)는 동일하다. 이 거리 함수를 사용하여 Nearest Neighbor 분류기를 CIFAR-10 데이터셋에 돌린다면, **35.4%** 정확도를 얻을 수 있다 (L1 거리를 사용한 결과보다 조금 낮아졌다).

**L1 vs. L2.** 두 거리 함수의 특징을 비교하는 것은 매우 흥미로운 주제이다. 일반적으로, L2 거리는 L1 거리에 비해 두 벡터간의 차가 커지는 것에 대해 훨씬 더 크게 반응한다. 즉, L2 거리는 하나의 큰 차이가 있는 것보다 여러 개의 적당한 차이가 생기는 것을 선호한다. L1/L2 거리(또는 두 이미지의 차에 대한 L1/L2 norm)는 일반적인 [p-norm](http://planetmath.org/vectorpnorm)의 형태 중 가장 많이 사용되는 두 가지이다.

<a name='knn'></a>

## k - Nearest Neighbor (kNN) 분류기

여태까지 예측을 할 때 가장 가까운 이미지의 라벨만을 사용하는 것을 이상하다고 생각할 수도 있을 것이다. 확실히, **k-Nearest Neighbor Classifier (kNN 분류기)** 라는 것을 사용한다면 거의 무조건 더 분류를 잘 할 수 있다. 아이디어는 매우 간단하다: 학습 데이터셋에서 가장 가까운 하나의 이미지만을 찾는 것이 아니라, 가장 가까운 **k** 개의 이미지를 찾아서 테스트 이미지의 라벨에 대해 투표하도록 하는 것이다. 여기서 *k = 1* 인 경우, 원래의 Nearest Neighbor 분류기가 된다. 직관적으로 **k** 값이 커질수록 분류기는 이상점(outlier)에 더 강인하고, 분류 경계가 부드러워지는 효과가 있다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/knn.jpeg">
  <div class="figcaption">Nearest Neighbor 분류기와 5-Nearest Neighbor 분류기의 차이 예시. 2차원 점과 3개의 클래스(라벨: red, blue, green)를 사용하였다. 색칠된 부분들은 L2 거리를 사용한 분류기를 통해 정해진 <b>결정 경계(decision boundaries)</b>이다. 흰색 부분들은 애매하게 분류(투표를 가장 많이 받은 라벨이 여러 개 있는 경우)된 점들을 나타낸다. NN 분류기의 경우 이상점들(e.g. 수많은 파란 점들 가운데에 있는 하나의 초록색 점)이 실제 결과와 맞지 않을 가능성이 큰 섬들을 형성하지만, 5-NN 분류기는 이런 조그만한 섬들이 생기지 않도록 부드럽게 이어주는 것을 확인하자. 이런 특성이 실제 테스트 데이터(그림에는 없음)에 적용할 때는 더 나은 <b>일반화(generalization)</b> 성능을 보인다. 또한, 5-NN 분류기 결과에서 회색 부분들은 nearest neighbors 간의 투표에서 동점이 발생한 경우(e.g. 2개의 이웃이 red, 다음 2개가 blue, 마지막 이웃이 green)인 것을 확인하자.</div>
</div>

실제 문제에 적용할 경우, 대부분은 NN 분류기보다는 k-Nearest Neighbor (kNN) 분류기를 사용하고 싶을 것이다. 그러나 어떤 *k* 값을 골라야 할까? 이 문제에 대해 지금부터 다룰 것이다.

<a name='val'></a>

### Hyperparameter 튜닝을 위한 검증 셋 (Validation set)

k-nearest neighbor 분류기는 *k* 를 정해줘야 한다. 그런데 어떤 값이 가장 좋을까? 또한, 앞서 우리는 여러 가지 거리 함수(L1 norm, L2 norm, 여기서 고려하지 않은 다른 종류들 - e.g.내적 - 도 매우 많다)에 대해서도 살펴보았다. 이러한 선택들을 **hyperparameters** 라 부르고, 데이터로부터 학습하는 많은 기계학습(머신러닝) 알고리즘 디자인에 등장한다. 그런데 어떤 값/세팅을 골라야 하는지에 대해서 확신이 있는 경우는 거의 없다.

여러 가지 다른 값들을 시도해보고, 어떤 것이 가장 좋은 성능을 보이는지 확인해보는 방법을 생각할 수 있다. 아래에서 우리도 실제로 이렇게 할 것이지만, 이 과정은 매우 조심스럽게 수행되어야 한다. 특히, **hyperparameter 값을 조정하기 위해 테스트 셋을 사용하면 절대 안 된다**. 우리가 머신러닝 알고리즘을 디자인할 때, 테스트 셋은 매우 귀한 리소스이고, 이론적으로는 실제로 알고리즘을 평가할 때인 맨 마지막 단 한 번을 제외하고는 절대 쳐다봐서는 안 된다. 그렇게 하지 않는다면 위험한 점은, 우리 모델의 hyperparameter 들이 테스트 셋에서는 잘 동작하도록 튜닝이 되어 있지만, 실전에서 모델을 사용(deploy)할 때 상당히 성능이 낮아지는 것을 확인할 수 있을 것이다. 머신러닝에서는 이것을 테스트 셋에 **overfit** 되었다고 말한다. 이를 다른 관점으로 바라본다면, 우리가 테스트 셋을 사용하여 hyperparameter 들을 튜닝했다는 것은 곧 우리가 테스트 셋을 마치 학습 데이터셋(트레이닝 셋)처럼 사용한 것이고, 우리 모델의 테스트 셋에서의 성능은 실제로 다른 데이터에 적용할 때에 비해 너무 낙관적이게 되어버린다. 그러나 테스트 셋을 맨 마지막에 딱 한 번만 사용한다면, 그 때는 우리가 학습한 분류기의 **일반화(generalization)** 된 성능을 잘 평가할 수 있는 척도로 활용될 것이다. (이 수업의 나중 부분에서도 일반화에 관련된 주제를 다룰 것이다.)

> 테스트 셋에 성능을 평가하는 것은 맨 마지막에 단 한 번만 하라.

다행히도, hyperparameter 들을 튜닝하는 올바른 방법이 존재하고, 이 방법은 테스트 셋을 전혀 건드리지 않는다. 아이디어는, 우리가 갖고 있는 트레이닝 셋을 두 개로 쪼개는 것이다: 이른바 **검증 셋(validation set)** 으로 불리는, 약간 적은 수의 트레이닝 셋과 나머지로 나눈다. CIFAR-10 데이터셋을 예로 들면, 학습 이미지들 중에 49,000 장을 트레이닝 셋으로 삼고, 나머지 1,000 개를 검증(validation) 용으로 남겨놓는 것이다. 이 검증 셋은 hyperparameter 들을 튜닝할 때, 가짜 테스트 셋으로 활용된다. (역자 주: 즉, 실전 테스트인 수능을 준비하기 위한 모의고사라고 생각하면 된다.)

CIFAR-10의 경우, 이런 식으로 나타낼 수 있을 것이다:

~~~python
# Xtr_rows, Ytr, Xte_rows, Yte 는 이전과 동일하게 갖고 있다고 가정하자.
# Xtr_rows 는 50,000 x 3072 행렬이었다.
Xval_rows = Xtr_rows[:1000, :] # 앞의 1000 개를 검증용으로 선택한다.
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # 뒤쪽의 49,000 개를 학습용으로 선택한다.
Ytr = Ytr[1000:]

# 검증 셋에서 가장 잘 동작하는 hyperparameter 들을 찾는다.
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # 특정 k 값을 정해서 검증 데이터에 대해 평가할 때 사용한다.
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # 여기서는 k를 input으로 받을 수 있도록 변형된 NearestNeighbor 클래스가 있다고 가정하자.
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # 검증 셋에 대한 정확도를 저장해 놓는다.
  validation_accuracies.append((k, acc))
~~~

이 과정이 끝나면, 어떤 *k* 값이 가장 잘 동작하는지를 그래프로 그려볼 수 있다. 그 뒤, 가장 잘 동작하는 k 값으로 정하고, 실제 테스트 셋에 대해 한 번 평가를 하면 된다.

> 학습 데이터셋을 트레이닝 셋과 검증 셋으로 나누고, 검증 셋을 활용하여 모든 hyperparameter 들을 튜닝하라. 마지막으로 테스트 셋에 대해서는 딱 한 번 돌려보고, 성능을 리포트한다.

**Cross-validation (교차 검증)**.
학습 데이터셋의 크기가 작을 경우(검증 셋의 크기도 작을 것이다), 조금 더 정교한 방식으로 **교차 검증(cross-validation)** 이라는 hyperparameter 튜닝 방법을 사용한다. 앞의 예시에서처럼 첫 1000 개의 데이터를 검증 셋으로 사용하고 나머지를 학습(training) 셋으로 사용하는 대신, 어떤 *k* 값이 더 좋은지를 여러 가지 검증 셋에 대해 시험해보고 평균 성능을 확인해본다면 보다 잡음이 덜 섞이고 나은 예측을 할 수 있을 것이다. 예를 들어, 5-fold 교차 검증에서는 학습 데이터를 5개의 동일한 크기의 그룹(fold)으로 쪼갠 뒤, 4개를 학습용으로, 1개를 검증용으로 사용한다. 그 다음에는 어떤 그룹을 검증 셋으로 사용할지에 따라 iteration(반복)을 돌고, 성능을 평가하고, 각 그룹에 대해 평가한 성능을 평균낸다.

<div class="fig figleft fighighlight">
  <img src="{{site.baseurl}}/assets/cvplot.png">
  <div class="figcaption">파라미터 <b>k</b> 에 대한 5-fold 교차 검증 예시. 각 <b>k</b> 값마다 4개의 그룹에 대해 학습을 하고 다섯 번째 그룹을 사용하여 성능을 평가한다. 따라서, 각 <b>k</b> 마다 검증 셋으로 활용한 그룹들에서 5 개의 정확도가 나온다. (y축이 정확도를 나타내고, 각 결과는 점으로 표시하였다.) 그래프에서 선은 각 <b>k</b> 에서의 결과의 평균으로 그려져 있고, 에러 바는 표준 편차를 나타낸다. 이 경우, 이 데이터셋에 대해서는 <b>k</b> = 7 로 놓는 것이 가장 좋을 것(그래프에서 가장 높은 부분)이라고 교차 검증 결과가 말해준다. 만약 5개보다 더 많은 그룹 수를 사용했다면, 지금보다는 더 부드러운 곡선 형태(즉, 잡음이 덜 섞여있음)의 그래프를 볼 수 있을 것이다.</div>
  <div style="clear:both"></div>
</div>


**실제 활용**. 교차 검증은 계산량이 매우 많아지기 때문에, 실제로 사람들은 교차 검증보다 하나의 검증 셋을 정해놓는 것을 선호한다. 보통은 학습 데이터의 50% ~ 90% 정도를 학습 용으로 쓰고 나머지를 검증 데이터로 활용하는데, 검증 데이터셋의 크기는 여러 가지 변수들에 의해 영향을 받는다. 예를 들어, hyperparameter 개수가 매우 많다면, 검증 데이터셋의 크기를 늘리는게 좋을 것이다. 검증 셋에 있는 데이터의 개수가 매우 적다면 (수백 개 정도), 교차 검증 방법을 사용하는 것이 더 안전하다. 보통은 3-fold, 5-fold, 10-fold 교차 검증을 주로 많이 사용한다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/crossval.jpeg">
  <div class="figcaption">데이터를 그룹으로 나누는 일반적인 방법. 학습 데이터셋과 테스트 셋은 주어져 있다. 학습 셋은 이 예시의 경우, 5개의 그룹으로 나누어져 있다. 이 중 1-4 그룹이 학습 셋이 되고, 나머지 하나(노란색 그룹 5)가 검증 셋으로 사용할 그룹으로 hyperparameter 들을 튜닝하는데 사용된다. 교차 검증 방법은 여기서 한 단계 더 나아가서 어떤 그룹을 검증 셋으로 사용할지를 1-5까지 바꿔가며 전부 반복하고, 이를 5-fold 교차 검증이라 부른다. 모델의 학습이 끝나고 가장 좋은 hyperparameter 들이 정해진 이후에는, 마지막으로 모델을 테스트 데이터(빨간색)에 대해 딱 한 번 시험해보고 성능을 평가한다.</div>
</div>

<a name='procon'></a>

**Nearest Neighbor 분류기의 장단점.**

Nearest Neighbor 분류기의 장점과 단점이 무엇인지 분석해보자. 당연히, 한 가지 장점은 방법을 이해하고 구현하는 것이 매우 쉽다는 점이다. 또한, 분류기를 학습할 때 단순히 학습 데이터셋을 저장하고 기억만 해놓으면 되기 때문에 학습 시간이 전혀 소요되지 않는다. 그러나, 학습 시의 계산량이 없는 것은 테스트할 때 모든 학습 데이터 예시들과 비교를 해야되기 때문에 계산량이 매우 많아지는 것으로 보상된다. 이것은 거꾸로인게, 보통 우리는 테스트할 때 얼마나 효율적인지에 관심이 많이 있고, 학습에 소요되는 시간이 얼마인지는 크게 중요하게 생각하지 않기 때문이다. 사실, 이 수업에서 나중에 다룰 (깊은) 뉴럴 네트워크, 또는 신경망 구조는 이 교환(tradeoff)을 반대 극단으로 이끈다. 뉴럴 네트워크는 학습할 때 매우 많은 계산량을 필요로 하지만, 학습이 끝나면 새로운 테스트 샘플을 분류하는데 매우 적은 계산만으로도 수행할 수 있다. 실제 환경에서는 이러한 형태가 더 바람직하다.

딴 얘기지만, Nearest Neighbor 분류기의 계산량(computational complexity) 문제는 매우 활발한 연구 주제이고, 많은 **Approximate Nearest Neighbor** (ANN, 근사 최근접 이웃) 알고리즘 및 라이브러리들이 있어서 데이터셋 내에서 nearest neighbor를 찾는 것을 가속화해준다 (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/)). 이 알고리즘들은 nearest neighbor를 찾는 것의 정확도를 조금 희생하여 공간(메모리)/시간(계산량) 복잡도를 크게 낮추도록 하고, 보통 kdtree나 k-means 알고리즘 등과 같은 전처리 기법에 의존하는 경우가 많다.

Nearest Neighbor 분류기가 좋은 경우도 있지만 (특히 데이터의 차원이 낮을 때), 실제 이미지 분류 문제 세팅에서는 대부분 효과적이지 않다. 한 가지 문제는, 이미지가 매우 고차원 물체라는 것이고 (수많은 픽셀들로 이루어져 있다), 고차원 공간에서의 '거리'는 매우 직관적이지 않는 경우가 많다. 아래 그림을 보면, 사람이 보기에 비슷한 이미지로 느끼는 것과 위에서 살펴본 픽셀 값들의 L2 거리를 기준으로 비슷한 것은 매우 다르다는 것을 알 수 있다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/samenorm.png">
  <div class="figcaption">고차원 데이터(이미지)에서의 픽셀값 기준 거리는 매우 비직관적인 경우가 많다. 원본 이미지(왼쪽)와 그 옆의 세 이미지는 픽셀값의 L2 거리를 기준으로 모두 같은 거리만큼 떨어져 있다. 이로 보아 픽셀값을 기준으로 한 거리는 인지적, 의미적으로 거의 연관이 없다고 생각할 수 있다.</div>
</div>

아래는 픽셀값의 차이만으로는 불충분하다는 점을 다시 한 번 보여주기 위한 시각화이다. 여기서는 <a href="http://homepage.tudelft.nl/19j49/t-SNE.html">t-SNE</a> 라는 시각화 기법을 사용하여 CIFAR-10 이미지들을 서로간의 거리가 잘 보존되도록 2차원으로 투사시킨 것이다. 이 시각화에서, 가까이 있는 이미지들은 픽셀간의 L2 거리가 매우 가까울 것이라고 생각하면 된다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/pixels_embed_cifar10.jpg">
  <div class="figcaption">t-SNE로 2차원으로 투사시킨 CIFAR-10 이미지들. 여기서 서로 가까이 있는 이미지들은 픽셀간의 L2 거리가 가까울 것이라고 생각하면 된다. 실제 클래스의 의미적인 차이보다 배경이 끼치는 영향이 얼마나 큰지 확인할 수 있다. 시각화의 큰 버전은 <a href="{{site.baseurl}}/assets/pixels_embed_cifar10_big.jpg">여기</a> 에서 확인할 수 있다.</div>
</div>

여기서 특히, 서로 가까운 이미지들은 보편적인 색의 분포나 배경의 종류에 영향을 많이 받고 각자의 실제 의미가 담긴 클래스에는 큰 영향을 받지 않는 것을 확인할 수 있다. 예를 들어, 강아지와 개구리가 똑같이 흰 배경에 있어서 (실제 클래스가 다름에도 불구하고) 매우 가까이 위치한 것을 볼 수 있다. 이상적으로는 같은 클래스의 이미지들이 여러 변칙적인 성질과 변화(또는 배경)에 상관없이 가까이 있어서 10개의 클래스들이 각각 군집을 이뤄서 뭉쳐있었으면 좋겠지만, 이러한 성질을 위해서는 단순 픽셀값 이상의 것이 필요하다.

<a name='summary'></a>

### 요약

- 여기서는 **이미지 분류(Image Classification)** 문제에 대해 살펴보았다. 각 이미지별로 한 개의 카테고리로 라벨링 되어있는 이미지들이 주어지고, 새로운 테스트 이미지들이 들어왔을 때 이 카테고리들 중 하나로 분류하도록 하고 예측값들의 정확도를 측정하였다.
- 간단한 **Nearest Neighbor 분류기** 를 소개하였다. 이 분류기와 관련하여 여러 가지 hyperparameter (k의 값이라든지, 데이터를 비교할 때 사용하는 거리의 종류라든지) 들이 존재하는 것을 보았고, 어떤 것을 선택할지 확실한 답은 없다는 것을 보았다.
- 이 hyperparameter 들을 올바르게 정하는 방법은 학습 데이터셋을 두 개로 (학습 셋과 **검증 셋(validation set)** 으로 불리는 가까 테스트 셋) 나누는 것임을 배웠다. 검증 셋에서 여러 가지 hyperparameter 값들을 시험해 보았고, 가장 좋은 성능을 얻는 값을 찾을 수 있었다.
- 학습 데이터가 적은 경우, 어떤 hyperparameter를 선택해야 하는지에 대해 보다 안정적인 방식인 **교차 검증(cross-validation)** 이라는 방법을 알게 되었다.
- 가장 좋은 hyperparameter 값들을 찾은 뒤, 그것으로 값을 고정하고 실제 테스트 셋에 대해 마지막에 단 한 번 **평가** 를 한다.
- Nearest Neighbor 분류기는 CIFAR-10 데이터셋에서 약 40% 정도의 정확도를 보이는 것을 확인하였다. 이 방법은 구현이 매우 간단하지만, 학습 데이터셋 전체를 메모리에 저장해야 하고, 새로운 테스트 이미지를 분류하고 평가할 때 계산량이 매우 많다.
- 마지막으로, 단순히 픽셀 값들의 L1이나 L2 거리는 이미지의 클래스보다 배경이나 이미지의 전체적인 색깔 분포 등에 더 큰 영향을 받기 때문에 이미지 분류 문제에 있어서 충분하지 못하다는 점을 보았다.

다음 강의에서는 여기서의 문제들을 해결하기 위한 방법들에 대해 살펴보고, 최종적으로 90% 정도의 성능을 갖고, 학습이 완료된 이후에는 학습 데이터셋을 전부 없애버려도 상관없으며, 테스트 이미지를 1/1000 초 단위로 빠르게 분류하고 평가할 수 있도록 해주는 모델을 살펴볼 것이다.

<a name='summaryapply'></a>

### 요약2: kNN을 실제로 적용하기

실제 응용에서 kNN을 사용하고 싶다면 (이미지에는 적용하지 않는 것을 추천하지만, 베이스라인으로 시도해볼 수는 있을 것이다), 다음 과정을 따르면 된다:

1. 데이터 전처리 과정을 수행하라: 데이터의 각 특징(feature)들을 평균이 0, 표준편차가 1이 되도록 정규화하라. 정규화 관련된 내용은 강의의 나중 부분에서도 다루겠지만, 여기서 따로 다루지 않았던 이유는 이미지의 픽셀들은 보통 균등한 분포를 갖기 때문에 데이터 정규화가 크게 필요하지 않기 때문이다.
2. 사용할 데이터가 매우 고차원 데이터라면, PCA ([wiki ref](http://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](http://www.bigdataexaminer.com/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/))나 아예 [Random Projection](http://scikit-learn.org/stable/modules/random_projection.html)과 같은 차원 축소 기법들을 적용하는 것을 고려해 보자.
3. 학습 데이터를 랜덤으로 학습/검증 셋(train/val split)으로 나누어라. 일반적으로, 70~90% 정도의 데이터를 학습용으로 사용한다. 이 세팅은 튜닝할 hyperparameter 들이 얼마나 많이 있는지에 따라, 각각이 얼만큼의 영향을 끼칠지에 따라 달라진다. 정해야 할 hyperparameter의 개수가 많다면, 그것들을 효과적으로 정하기 위해 충분히 큰 검증 셋을 사용해야 한다. 검증 셋의 크기가 적당한지에 대해서 의문이 있다면, 학습 데이터를 그룹으로 나누어서 교차 검증을 하는 방법이 제일 좋다. 계산할 시간만 충분하다면, 교차 검증을 하는 것이 항상 더 안전하다 (그룹이 많을수록 더 좋지만, 그만큼 계산량도 늘어난다).
4. 여러 가지 **k** 값에 대해 (많이 해볼수록 좋다), 다른 종류의 거리 함수에 대해 (L1과 L2 거리를 주로 사용한다) kNN 분류기를 학습하고, 검증 셋으로 (또는 교차 검증을 사용한다면 모든 그룹에 대해) 평가해 보자.
5. 현재의 kNN 분류기가 너무 느리다면, 이를 가속하기 위해 Approximate Nearest Neighbor 라이브러리 (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/))를 사용하는 것을 고려해보라. (성능은 조금 떨어질 것이다)
6. 가장 좋은 결과를 주는 hyperparameter 들을 기록해두라. 가장 좋은 hyperparameter 세팅으로 다시 전체 학습 데이터셋을 학습해야 하는지에 대한 점은 아직 확실하지 않다. 학습 셋에서 쪼갠 검증 셋을 다시 합친다면 최적의 hyperparameter 세팅이 바뀔 수도 있기 때문이다 (학습에 사용한 데이터셋의 크기가 커지기 때문에). 실제로는 최종 분류기에서 검증 셋은 사용하지 않는 편이 더 깔끔하고, 검증에 사용한 데이터들은 hyperparameter 들을 고르는데 사용되어 *날라가버렸다* 고 생각해도 된다. 그 뒤, 최종 모델을 테스트 셋에 대해 성능을 평가해 보고, 그 테스트 셋에 대한 정확도를 현재 데이터로 학습한 kNN 분류기의 성능으로 발표하라.

<a name='reading'></a>

#### 추가 읽기 자료

관심있을 법한 추가적인 읽기 자료 몇 가지를 선정해 두었다 (optional):

- [A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) 에서 section 6이 가장 연관이 있지만, 전체적인 내용을 다 읽는 것도 추천한다.

- [Recognizing and Learning Object Categories](http://people.csail.mit.edu/torralba/shortCourseRLOC/index.html). 물체 분류에 관한 ICCV 2005 (컴퓨터비전 분야에서 유명한) 학회의 short course.

---
<p style="text-align:right"><b>
번역: 이옥민 <a href="https://github.com/OkminLee" style="color:black">(OkminLee)</a>,
  최명섭<a href="https://github.com/myungsub" style="color:black">(myungsub)</a>
</b></p>
