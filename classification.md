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
In cases where the size of your training data (and therefore also the validation data) might be small, people sometimes use a more sophisticated technique for hyperparameter tuning called **cross-validation**. Working with our previous example, the idea is that instead of arbitrarily picking the first 1000 datapoints to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of *k* works by iterating over different validation sets and averaging the performance across these. For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.

<div class="fig figleft fighighlight">
  <img src="{{site.baseurl}}/assets/cvplot.png">
  <div class="figcaption">Example of a 5-fold cross-validation run for the parameter <b>k</b>. For each value of <b>k</b> we train on 4 folds and evaluate on the 5th. Hence, for each <b>k</b> we receive 5 accuracies on the validation fold (accuracy is the y-axis, each result is a point). The trend line is drawn through the average of the results for each <b>k</b> and the error bars indicate the standard deviation. Note that in this particular case, the cross-validation suggests that a value of about <b>k</b> = 7 works best on this particular dataset (corresponding to the peak in the plot). If we used more than 5 folds, we might expect to see a smoother (i.e. less noisy) curve.</div>
  <div style="clear:both"></div>
</div>


**In practice**. In practice, people prefer to avoid cross-validation in favor of having a single validation split, since cross-validation can be computationally expensive. The splits people tend to use is between 50%-90% of the training data for training and rest for validation. However, this depends on multiple factors: For example if the number of hyperparameters is large you may prefer to use bigger validation splits. If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation. Typical number of folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/crossval.jpeg">
  <div class="figcaption">Common data splits. A training and test set is given. The training set is split into folds (for example 5 folds here). The folds 1-4 become the training set. One fold (e.g. fold 5 here in yellow) is denoted as the Validation fold and is used to tune the hyperparameters. Cross-validation goes a step further iterates over the choice of which fold is the validation fold, separately from 1-5. This would be referred to as 5-fold cross-validation. In the very end once the model is trained and all the best hyperparameters were determined, the model is evaluated a single time on the test data (red).</div>
</div>

<a name='procon'></a>

**Pros and Cons of Nearest Neighbor classifier.**

It is worth considering some advantages and drawbacks of the Nearest Neighbor classifier. Clearly, one advantage is that it is very simple to implement and understand. Additionally, the classifier takes no time to train, since all that is required is to store and possibly index the training data. However, we pay that computational cost at test time, since classifying a test example requires a comparison to every single training example. This is backwards, since in practice we often care about the test time efficiency much more than the efficiency at training time. In fact, the deep neural networks we will develop later in this class shift this tradeoff to the other extreme: They are very expensive to train, but once the training is finished it is very cheap to classify a new test example. This mode of operation is much more desirable in practice.

As an aside, the computational complexity of the Nearest Neighbor classifier is an active area of research, and several **Approximate Nearest Neighbor** (ANN) algorithms and libraries exist that can accelerate the nearest neighbor lookup in a dataset (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/)). These algorithms allow one to trade off the correctness of the nearest neighbor retrieval with its space/time complexity during retrieval, and usually rely on a pre-processing/indexing stage that involves building a kdtree, or running the k-means algorithm.

The Nearest Neighbor Classifier may sometimes be a good choice in some settings (especially if the data is low-dimensional), but it is rarely appropriate for use in practical image classification settings. One problem is that images are high-dimensional objects (i.e. they often contain many pixels), and distances over high-dimensional spaces can be very counter-intuitive. The image below illustrates the point that the pixel-based L2 similarities we developed above are very different from perceptual similarities:

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/samenorm.png">
  <div class="figcaption">Pixel-based distances on high-dimensional data (and images especially) can be very unintuitive. An original image (left) and three other images next to it that are all equally far away from it based on L2 pixel distance. Clearly, the pixel-wise distance does not correspond at all to perceptual or semantic similarity.</div>
</div>

Here is one more visualization to convince you that using pixel differences to compare images is inadequate. We can use a visualization technique called <a href="http://homepage.tudelft.nl/19j49/t-SNE.html">t-SNE</a> to take the CIFAR-10 images and embed them in two dimensions so that their (local) pairwise distances are best preserved. In this visualization, images that are shown nearby are considered to be very near according to the L2 pixelwise distance we developed above:

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/pixels_embed_cifar10.jpg">
  <div class="figcaption">CIFAR-10 images embedded in two dimensions with t-SNE. Images that are nearby on this image are considered to be close based on the L2 pixel distance. Notice the strong effect of background rather than semantic class differences. Click <a href="{{site.baseurl}}/assets/pixels_embed_cifar10_big.jpg">here</a> for a bigger version of this visualization.</div>
</div>

In particular, note that images that are nearby each other are much more a function of the general color distribution of the images, or the type of background rather than their semantic identity. For example, a dog can be seen very near a frog since both happen to be on white background. Ideally we would like images of all of the 10 classes to form their own clusters, so that images of the same class are nearby to each other regardless of irrelevant characteristics and variations (such as the background). However, to get this property we will have to go beyond raw pixels.

<a name='summary'></a>

### Summary

In summary:

- We introduced the problem of **Image Classification**, in which we are given a set of images that are all labeled with a single category. We are then asked to predict these categories for a novel set of test images and measure the accuracy of the predictions.
- We introduced a simple classifier called the **Nearest Neighbor classifier**. We saw that there are multiple hyper-parameters (such as value of k, or the type of distance used to compare examples) that are associated with this classifier and that there was no obvious way of choosing them.
-  We saw that the correct way to set these hyperparameters is to split your training data into two: a training set and a fake test set, which we call **validation set**. We try different hyperparameter values and keep the values that lead to the best performance on the validation set.
- If the lack of training data is a concern, we discussed a procedure called **cross-validation**, which can help reduce noise in estimating which hyperparameters work best.
- Once the best hyperparameters are found, we fix them and perform a single **evaluation** on the actual test set.
- We saw that Nearest Neighbor can get us about 40% accuracy on CIFAR-10. It is simple to implement but requires us to store the entire training set and it is expensive to evaluate on a test image.
- Finally, we saw that the use of L1 or L2 distances on raw pixel values is not adequate since the distances correlate more strongly with backgrounds and color distributions of images than with their semantic content.

In next lectures we will embark on addressing these challenges and eventually arrive at solutions that give 90% accuracies, allow us to completely discard the training set once learning is complete, and they will allow us to evaluate a test image in less than a millisecond.

<a name='summaryapply'></a>

### Summary: Applying kNN in practice

If you wish to apply kNN in practice (hopefully not on images, or perhaps as only a baseline) proceed as follows:

1. Preprocess your data: Normalize the features in your data (e.g. one pixel in images) to have zero mean and unit variance. We will cover this in more detail in later sections, and chose not to cover data normalization in this section because pixels in images are usually homogeneous and do not exhibit widely different distributions, alleviating the need for data normalization.
2. If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA ([wiki ref](http://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](http://www.bigdataexaminer.com/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/)) or even [Random Projections](http://scikit-learn.org/stable/modules/random_projection.html).
3. Split your training data randomly into train/val splits. As a rule of thumb, between 70-90% of your data usually goes to the train split. This setting depends on how many hyperparameters you have and how much of an influence you expect them to have. If there are many hyperparameters to estimate, you should err on the side of having larger validation set to estimate them effectively. If you are concerned about the size of your validation data, it is best to split the training data into folds and perform cross-validation. If you can afford the computational budget it is always safer to go with cross-validation (the more folds the better, but more expensive).
4. Train and evaluate the kNN classifier on the validation data (for all folds, if doing cross-validation) for many choices of **k** (e.g. the more the better) and across different distance types (L1 and L2 are good candidates)
5. If your kNN classifier is running too long, consider using an Approximate Nearest Neighbor library (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/)) to accelerate the retrieval (at cost of some accuracy).
6. Take note of the hyperparameters that gave the best results. There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be *burned* on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.

<a name='reading'></a>

#### Further Reading

Here are some (optional) links you may find interesting for further reading:

- [A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf), where especially section 6 is related but the whole paper is a warmly recommended reading.

- [Recognizing and Learning Object Categories](http://people.csail.mit.edu/torralba/shortCourseRLOC/index.html), a short course of object categorization at ICCV 2005.

---
<p style="text-align:right"><b>
번역: 이옥민 <a href="https://github.com/OkminLee" style="color:black">(OkminLee)</a>,
  최명섭<a href="https://github.com/myungsub" style="color:black">(myungsub)</a>
</b></p>
