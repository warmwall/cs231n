---
layout: page
permalink: /optimization-1/
---

목자:

- [소개](#intro)
- [손실함수(Loss Function)의 시각화(Visualization)](#vis)
- [최적화(Optimization)](#optimization)
  - [전략 #1: 무작위 탐색 (Random Search)](#opt1)
  - [전략 #2: 무작위 국소 탐색 (Random Local Search)](#opt2)
  - [전략 #3: 그라디언트(gradient) 따라가기](#opt3)
- [그라디언트(Gradient) 계산](#gradcompute)
  - [Finite Differences를 이용한 수치적인 방법](#numerical)
  - [미분을 이용한 해석적인 방법](#analytic)
- [그라디언트 하강(Gradient Descent)](#gd)
- [요약](#summary)

<a name='intro'></a>

### 소개

이전 섹션에서 이미지 분류(image classification)을 할 때에 있어 두 가지의 핵심요쇼를 소개했습니다.

1. 원 이미지의 픽셀들을 넣으면 분류 스코어(class score)를 계산해주는 파라미터화된(parameterized) **스코어함수(score function)** (예를 들어,  선형함수).
2. 학습(training) 데이타에 어떤 특정 파라미터(parameter/weight)들을 가지고 스코어함수(score function)를 적용시켰을 때, 실제 class와 얼마나 잘 일치하는지에 따라 그 특정 파라미터(parameter/weight)들의 질을 측정하는 **손실함수(loss function)**. 여러 종류의 손실함수(예를 들어, Softmax/SVM)가 있다.

구체적으로 말하자면, 다음과 같은 형식을 가진 선형함수 $$ f(x_i, W) =  W x_i $$를 스코어함수(score function)로 쓸 때,  앞에서 다룬 바와 같이 SVM은 다음과 같은 수식으로 표현할 수 있다.:

$$
L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + 1) \right] + \alpha R(W)
$$

예시 $x_i$에 대한 예측값이 실제 값(레이블, labels) $$y_i$$과 같도록 설정된 파라미터(parameter/weight) $$W$$는 손실(loss)값 $$L$$ 또한 매우 낮게 나온다는 것을 알아보았다. 이제 세번째이자 마지막 핵심요소인 **최적화(optimization)**에 대해서 알아보자. 최적화(optimization)는 손실함수(loss function)을 최소화시카는 파라미터(parameter/weight, $$W$$)들을 찾는 과정을 뜻한다.

**예고:** 이 세 가지 핵심요소가 어떻게 상호작용하는지 이해한 후에는, 첫번째 요소(파라미터화된 함수)로 다시 돌아가서 선형함수보다 더 복잡한 형태로 확장시켜볼 것이다.  처음엔 신경망(Neural Networks), 다음엔 컨볼루션 신경망(Convolutional Neural Networks). 손실함수(loss function)와 최적화(optimization) 과정은 거의 변화가 없을 것이다..

<a name='vis'></a>

### 손실함수(loss function)의 시각화

이 강의에서 우리가 다루는 손실함수(loss function)들은 대체로 고차원 공간에서 정의된다. 예를 들어, CIFAR-10의 선형분류기(linear classifier)의 경우 파라미터(parameter/weight) 행렬은 크기가 [10 x 3073]이고 총 30,730개의 파라미터(parameter/weight)가 있다. 따라서, 시각화하기가 어려운 면이 있다. 하지만, 고차원 공간을 1차원 직선이나 2차원 평면으로 잘라서 보면 약간의 직관을 얻을 수 있다. 예를 들어, 무작위로 파라미터(parameter/weight) 행렬 $W$을 하나 뽑는다고 가정해보자. (이는 사실 고차원 공간의 한 점인 셈이다.) 이제 이 점을 직선 하나를 따라 이동시키면서 손실함수(loss function)를 기록해보자. 즉, 무작위로 뽑은 방향 $$W_1$$을 잡고, 이 방향을 따라 가면서 손실함수(loss function)를 계산하는데, 구체적으로 말하면 $$L(W + a W_1)$$에 여러 개의 $$a$$ 값(역자 주: 1차원 스칼라)을 넣어 계산해보는 것이다. 이 과정을 통해 우리는 $$a$$ 값을 x축, 손실함수(loss function) 값을 y축에 놓고 간단한 그래프를 그릴 수 있다. 또한 이 비슷한 것을 2차원으로도 할 수 있다. 여러 $$a, b$$값에 따라  $$ L(W + a W_1 + b W_2) $$을 계산하고(역자 주: $$W_2$$ 역시 $$W_1$$과 같은 식으로 뽑은 무작위 방향), $$a, b$$는 각각 x축과 y축에, 손실함수(loss function) 값 색을 이용해 그리면 된다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/svm1d.png">
  <img src="{{site.baseurl}}/assets/svm_one.jpg">
  <img src="{{site.baseurl}}/assets/svm_all.jpg">
  <div class="figcaption">
    Regularization 없는 멀티클래스 SVM의 손실함수(Loss function)의 지형을 CIFAR-10 데이타의 1개의 예시(왼쪽, 가운데)와 여러 개의 예시(오른쪽)에 적용시켜 그려본 그림들. 왼쪽: 여러 <b>a</b>값에 따른 1차원 손실(loss) 곡선. 가운데, 오른쪽: 2차원 손실(loss) 평면, 파란색은 낮은 손실(loss)를 뜻하고, 빨간색은 높은 손실(=loss)를 뜻한다. 손실함수(Loss function)가 부분적으로 선형(piecewise linear)인 것이 특징이다. 특히, 오른쪽 그림은 여러 예시를 통해 구한 손실(loss)들을 평균낸 것인데, 밥공기 모양인 것이 특징이다. 이는 가운데 그림 같은 각진 모양의 밥공기 여러 개를 평균낸 모양인 셈이다.
  </div>
</div>

부분적으로 선형(piecewise linear)은 손실함수(Loss function)의 구조를 수식을 통해 설명할 수 있다. 예시가 하나인 경우에 다음과 같이 쓸 수 있다.

$$
L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + 1) \right]
$$

수식에서 명백히 볼 수 있듯이, 각 예시의 손실(loss)값은 ($$\max(0,-)$$ 함수로 인해 0에서 막혀있는) $$W$$의 선형함수들의 합으로 표현된다. $$W$$의 각 행(즉, $$w_j$$) 앞에는 때때로 (잘못된 분류일 때, 즉, $$j\neq y_i$$인 경우) 플러스가 붙고, 때때로 (옳은 분류일 때) 마이너스가 붙는다. 더 명확히 표현하자면, 3개의 1차원 점들과 3개의 클래스가 있다고 해보자. Regularization 없는 총 SVM 손실(loss)은 다음과 같다.

$$
\begin{align}
L_0 = & \max(0, w_1^Tx_0 - w_0^Tx_0 + 1) + \max(0, w_2^Tx_0 - w_0^Tx_0 + 1) \\\\
L_1 = & \max(0, w_0^Tx_1 - w_1^Tx_1 + 1) + \max(0, w_2^Tx_1 - w_1^Tx_1 + 1) \\\\
L_2 = & \max(0, w_0^Tx_2 - w_2^Tx_2 + 1) + \max(0, w_1^Tx_2 - w_2^Tx_2 + 1) \\\\
L = & (L_0 + L_1 + L_2)/3
\end{align}
$$

이 예시들이 1차원이기 때문에, 데이타 $$x_i$$와 파라미터(parameter/weight) $$w_j$$는 숫자(역자 주: 즉, 스칼라. 따라서 위 수식에서 전치행렬을 뜻하는 $$T$$ 표시는 필요없음)이다. 예를 들어 $$w_0$$ 를 보면, 몇몇 항들은 $$w_0$$의 선형함수이고 각각은 0에서 꺾인다. 이를 다음과 같이 시각화할 수 있다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/svmbowl.png">
  <div class="figcaption">
    손실(loss)를 1차원으로 표현한 그림. x축은 파라미터(parameter/weight) 하나이고, y축은 손실(loss)이다. 손실(loss)는 여러 항들의 합인데, 그 각각은 특정 파라미터(parameter/weight)값과 무관하거나, 0에 막혀있는 그 파라미터(parameter/weight)의 선형함수이다. 전체 SVM 손실은 이 모양의 30,730차원 버전이다.
  </div>
</div>

옆길로 새면, 아마도 밥공기 모양을 보고 SVM 손실함수(loss function)이 일종의 [볼록함수](http://en.wikipedia.org/wiki/Convex_function)라고 생각했을 것이다. 이런 형태의 함수를 효율적으로 최소화하는 문제에 대한 엄청난 양의 연구 성과들이 있다. 스탠포드 강좌 중에서도 이 주제를 다룬 것도 있다. ( [볼록함수 최적화](http://stanford.edu/~boyd/cvxbook/) ). 이 스코어함수(score function) $$f$$를 신경망(neural networks)로 확장시키면, 목적함수(역자 주: 손실함수(loss function))은 더이상 볼록함수가 아니게 되고, 위와 같은 시각화를 해봐도 밥공기 모양 대신 울퉁불퉁하고 복잡한 모양이 보일 것이다.

*미분이 불가능한 손실함수(loss functions)*. 기술적인 설명을 덧붙이자면, $$\max(0,-)$$ 함수 때문에 손실함수(loss functionn)에 *꺾임*이 생기는데, 이 때문에 손실함수(loss functions)는 미분이 불가능해진다. 왜냐하면, 그 꺾이는 부분에서 미분 혹은 그라디언트가 존재하지 않기 때문이다. 하지만, [서브그라디언트(subgradient)](http://en.wikipedia.org/wiki/Subderivative)가 존재하고, 대체로 이를 그라디언트(gradient) 대신 이용한다. 앞으로 이 강의에서는 *그라디언트(gradient)*와 *서브그라디언트(subgradient)*를 구분하지 않고 쓸 것이다.

<a name='최적화'></a>

### 최적화

정리하면, 손실함수(loss function)는 파라미터(parameter/weight) **W** 행렬의 질을 측정한다. 최적화의 목적은 이 손실함수(loss function)을 최소화시키는 **W**을 찾아내는 것이다. 다음 단락부터 손실함수(loss function)을 최적화하는 방법에 대해서 찬찬히 살펴볼 것이다. 이전에 경험이 있는 사람들이 보면 이 섹션은 좀 이상하다고 생각할지 모르겠다. 왜냐하면, 여기서 쓰인 예 (즉, SVM 손실(loss))가 볼록함수이기 때문이다. 하지만, 우리의 궁극적인 목적은 신경망(neural networks)를 최적화시키는 것이고, 여기에는 볼록함수 최적화를 위해 고안된 방법들이 쉽사리 통히지 않는다.

<a name='opt1'></a>

#### 전략 #1: 첫번째 매우 나쁜 방법: 무작위 탐색 (Random search)

주어진 파라미터(parameter/weight) **W**이 얼마나 좋은지를 측정하는 것은 매우 간단하기 때문에, 처음 떠오르는 (매우 나쁜) 생각은, 단순히 무작위로 파라미터(parameter/weight)을 골라서 넣어보고 넣어 본 값들 중 제일 좋은 값을 기록하는 것이다. 그 과정은 다음과 같다.

~~~python
# X_train의 각 열(column)이 예제 하나에 해당하는 행렬이라고 생각하자. (예를 들어, 3073 x 50,000짜리)
# Y_train 은 레이블값이 저장된 어레이(array)이라고 하자. (즉, 길이 50,000짜리 1차원 어레이)
# 그리고 함수 L이 손실함수라고 하자.

bestloss = float("inf") # Python assigns the highest possible float value
for num in xrange(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

# prints:
# in attempt 0 the loss was 9.401632, best 9.401632
# in attempt 1 the loss was 8.959668, best 8.959668
# in attempt 2 the loss was 9.044034, best 8.959668
# in attempt 3 the loss was 9.278948, best 8.959668
# in attempt 4 the loss was 8.857370, best 8.857370
# in attempt 5 the loss was 8.943151, best 8.857370
# in attempt 6 the loss was 8.605604, best 8.605604
# ... (trunctated: continues for 1000 lines)
~~~

위의 코드에서, 여러 개의 무작위 파라미터(parameter/weight) **W**를 넣어봤고, 그 중 몇몇은 다른 것들보다 좋았다. 그래서 그 중 제일 좋은 파라미터(parameter/weight) **W**을 테스트 데이터에 넣어보면 된다.

~~~python
# X_test은 크기가 [3073 x 10000]인 행렬, Y_test는 크기가 [10000 x 1]인 어레이라고 하자.
scores = Wbest.dot(Xte_cols) # 모든 테스트데이터 예제(1만개)에 대한 각 클라스(10개)별 점수를 모아놓은 크기 10 x 10000짜리인 행렬
# 각 열(column)에서 가장 높은 점수에 해당하는 클래스를 찾자. (즉, 예측 클래스)
Yte_predict = np.argmax(scores, axis = 0)
# 그리고 정확도를 계산하자. (예측 성공률)
np.mean(Yte_predict == Yte)
# 정확도 값이 0.1555라고 한다.
~~~

이 방법으로 얻은 최선의 **W**는 정확도 **15.5%**이다. 완전 무작위 찍기가 단 10%의 정확도를 보이므로, 무식한 방법 치고는 그리 나쁜 것은 아니다.

**핵심 아이디어: 반복적 향상**. 물론 이보다 더 좋은 방법들이 있다. 여기서 핵심 아이디어는, 최선의 파라미터(parameter/weight) **W**을 찾는 것은 매우 어렵거나 때로는 불가능한 문제(특히 복잡한 신경망(neural network) 전체를 구현할 경우)이지만, 어떤 주어진 파라미터(parameter/weight) **W**을 조금 개선시키는 일은 훨씬 덜 힘들다는 점이다. 다시 말해, 우리의 접근법은 무작위로 뽑은 **W**에서 출발해서 매번 조금씩 개선시키는 것을 반복하는 것이다.

> 우리의 전략은 무작위로 뽑은 파라미터(parameter/weight)으로부터 시작해서 반복적으로 조금씩 개선시켜 손실(loss)을 낮추는 것이다.

**눈 가리고 하산하는 것에 비유.** 앞으로 도움이 될만한 비유는, 경사진 지형에서 눈가리개를 하고 점점 아래로 내려오는 자기 자신을 생각해보는 것이다. CIFAR-10의 예시에서, 그 언덕들은 (**W**가 3073 x 10 차원이므로) 30,730차원이다. 언덕의 각 지점에서의 고도가 손실함수(loss function)의 손실값(loss)의 역할을 한다.

<a name='opt2'></a>

#### 전략 #2: 무작위 국소 탐색 (Random Local Search)

처음 떠오르는 전략은, 시작점에서 무작위로 방향을 정해서 발을 살짝 뻗어서 더듬어보고 그게 내리막길길을 때만 한발짝 내딛는 것이다. 구체적으로 말하면, 임의의 $$W$$에서 시작하고, 또다른 임의의 방향 $$ \delta W $$으로 살짝 움직여본다. 만약에 움직여간 자리($$W + \delta W$$)에서의 손실잢(loss)가 더 낮으면, 거기로 움직이고 다시 탐색을 시작한다. 이 과정을 코드로 짜면 다음과 같다.

~~~python
W = np.random.randn(10, 3073) * 0.001 # 임의의 시작 파라미터를 랜덤하게 고른다.
bestloss = float("inf")
for i in xrange(1000):
  step_size = 0.0001
  Wtry = W + np.random.randn(10, 3073) * step_size
  loss = L(Xtr_cols, Ytr, Wtry)
  if loss < bestloss:
    W = Wtry
    bestloss = loss
  print 'iter %d loss is %f' % (i, bestloss)
~~~

이전과 같은 횟수(즉, 1000번)만큼 손실함수(loss function)을 계산하고도, 이 방법을 테스트 데이터에 적용해보니,  분류정확도가 **21.4%**로 나왔다. 발전하긴 했지만, 여전히 좀 비효울적인 것 같다.

<a name='opt3'></a>

#### 전략 #3: 그라디언트(gradient) 따라가기

이전 섹션에서, 파라미터(parameter/weight) 공간에서 파라미터(parameter/weight) 벡터를 향상시키는 (즉, 손실값을 더 낮추는) 뱡향을 찾는 시도를 해봤다. 그런데 사실 좋은 방향을 찾기 위해 방향을 무작위로 탐색할 필요가 없다고 한다. (적어도 반지름이 0으로 수렴하는 아주 좁은 근방에서는) 가장 가파르게 감소한다고 수학적으로 검증된 *최선의* 방향을 구할 수 있고, 이 방향을 따라 파라미터(parameter/weight) 벡터를 움직이면 된다는 것이다. 이 방향이 손실함수(loss function)의 **그라디언트(gradient)**와 관계있다. 눈 가리고 하산하는 것에 비유할 때, 발 밑 지형을 잘 더듬어보고 가장 가파르다는 느낌을 주는 방향으로 내려가는 것에 비견할 수 있다.

1차원 함수의 경우, 어떤 점에서 움직일 때 기울기는 함수값의 순간 증가율을 나타낸다. 그라디언트(gradient)는 이 기울기란 것을, 변수가 하나가 아니라 여러 개인 경우로 일반화시킨 것이다. 덧붙여 설명하면, 그라디언트(gradient)는 입력데이터공간(역자 주: x들의 공간)의 각 차원에 해당하는 기울기(**미분**이라고 더 많이 불린다)들의 백터이다. 1차원 함수의 미분을 수식으로 쓰면 다음과 같다.

$$
\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}
$$

함수가 숫자 하나가 아닌 벡터를 입력으로 받는 경우 (역자 주: x가 벡터인 경우), 우리는 미분을 **편미분**이라고 부른고, 그라디언트(gradient)는 단순히 각 차원으로의 편미분들을 모아놓은 벡터이다.

<a name='gradcompute'></a>

### 그라디언트(gradient) 계산

그라디언트(gradient) 계산법은 크게 2가지가 있다: 느리고 근사값이지만 쉬운 방법 (**수치 그라디언트**), and 빠르고 정확하지만 미분이 필요하고 실수하기 쉬운 방법 (**해석적 그라디언트**). 여기서 둘 다 다룰 것이다.

<a name='numerical'></a>

#### 유한한 차이(Finite Difference)를 이용하여 수치적으로 그라디언트(gradient) 계산하기

위에 주어진 수식을 이용하여 그라디언트(gradient)를 수치적으로 계산할 수 있다. 여기 임의의 함수 `f`, 이 함수에 입력값으로 넣을 벡터 `x` 가 주어졌을 때, `x`에서 `f`의 그라디언트(gradient)를 계산해주는 범용 함수가 있다:

~~~python
def eval_numerical_gradient(f, x):
  """
함수 f의 x에서의 그라디언트를 매우 단순하게 구현하기.
- f 는 입력값 1개를 받는 함수여야한다.
 - x는 numpy 어레이(array)로서그라디언트를 계산할 지점 (역자 주: 그라디언트는 당연하게도 어디서 계산하느냐에 따라 달라지므로, 함수 f 뿐 아니라 x도 정해줘야함).
  """

  fx = f(x) # 원래 지점 x에서 함수값 구하기.
  grad = np.zeros(x.shape)
  h = 0.00001

  # x의 모든 인덱스를 다 돌면서 계산하기.
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # 함수 값을  x+h에서 계산하기.
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # 변화랑h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # 이전 값을 다시 가져온다. (매우 중요!)

    # 편미분 계산
    grad[ix] = (fxh - fx) / h # 기울기
    it.iternext() # 다음 단계로 가서 반복.

  return grad
~~~

이 코드는, 위에 주어진 그라디언트(gradient) 식을 이용해서 모든 차원을 하나씩 돌아가면서 그 방향으로 작은 변화 `h`를 줬을 때, 손실함수(loss function)의 값이 얼마나 변하는지를 구해서, 그 방향의 편미분 값을 계산한다. 변수 `grad`에 전체 그라디언트(gradient) 값이 최종적으로 저장된다.

**실제 고려할 사항**. **h**가 0으로 수렴할 때의 극한값이 그라디언트(gradient)의 수학적으로 정의인데, (이 예시에서 나온 것처럼 1e-5 같이) 작은 값이면 충분하다. 이상적으로, 수치적인 문제를 일으키지 않는 수준에서 가장 작은 값을 쓰면 된다. 덧붙여서, 실제 활용할 때, x를 **양 방향으로 변화를 주어서 구한 수식**이 더 좋은 경우가 많다: $ [f(x+h) - f(x-h)] / 2 h $ . 다음  [위키](http://en.wikipedia.org/wiki/Numerical_differentiation)를 보면 자세한 것을 알 수 있다.

위에서 계산한 함수를 이용하면, 아무 함수의 아무 값에서나 그라디언트(gradient)를 계산할 수 있다. 무작위로 뽑은 파라미터(parameter/weight)값에서 CIFAR-10의 손실함수(loss function)의 그라디언트를 구해본다.:

~~~python

# 위의 범용코드를 쓰려면 함수가 입력값 하나(이 경우 파라미터)를 받아야함.
 # 따라서X_train와 Y_train은 입력값으로 안 치고 W 하나만 입력값으로 받도록 함수 다시 정의.
def CIFAR10_loss_fun(W):
  return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001 # 랜덤 파라미터 벡터.
df = eval_numerical_gradient(CIFAR10_loss_fun, W) # 그라디언트를 구했다.
~~~

그라디언트(gradient)는 각 차원에서 CIFAR-10의 손실함수(loss function)의 기울기를 알려주는데, 그걸 이용해서 파라미터(parameter/weight)를 업데이트한다.

~~~python
loss_original = CIFAR10_loss_fun(W) # 기존 손실값
print 'original loss: %f' % (loss_original, )

# 스텝크기가 주는 영향에 대해 알아보자.
for step_size_log in [-10, -9, -8, -7, -6, -5,-4,-3,-2,-1]:
  step_size = 10 ** step_size_log
  W_new = W - step_size * df # 파라미터(parameter/weight) 공간 상의 새 파라미터 값
  loss_new = CIFAR10_loss_fun(W_new)
  print 'for step size %f new loss: %f' % (step_size, loss_new)

# prints:
# original loss: 2.200718
# for step size 1.000000e-10 new loss: 2.200652
# for step size 1.000000e-09 new loss: 2.200057
# for step size 1.000000e-08 new loss: 2.194116
# for step size 1.000000e-07 new loss: 2.135493
# for step size 1.000000e-06 new loss: 1.647802
# for step size 1.000000e-05 new loss: 2.844355
# for step size 1.000000e-04 new loss: 25.558142
# for step size 1.000000e-03 new loss: 254.086573
# for step size 1.000000e-02 new loss: 2539.370888
# for step size 1.000000e-01 new loss: 25392.214036
~~~

**Update in negative gradient direction**. 위 코드에서, 새로운 파라미터 `W_new`로 업데이트할 때, 그라디언트(gradient) `df`의 반대방향으로 움직인 것을 주목하자. 왜냐하면 우리가 원하는 것은 손실함수(loss function)의 증가가 아니라 감소하는 것이기 때문이다.

**스텝 크기가 미치는 영향**. 그라디언트(gradient)에서 알 수 있는 것은 함수값이 가장 빠르게 증가하는 방향이고, 그 방향으로 대체 얼만큼을 가야하는지는 알려주지 않는다. 강의 뒤에서 다루게 되겠지만, 얼만큼 가야하는지를 의미하는 스텝 크기(혹은 *학습 속도*라고도 함)는 신경망(neural network)를 학습시킬 때 있어 가장 중요한 (그래서 결정하기 까다로운) 하이퍼파라미터(hyperparameter)가 될 것이다. 눈 가리고 하산하는 비유에서, 우리는 우리 발 밑으로 어느 방향이 가장 가파른지 느끼지만, 얼마나 발을 뻗어야할 지는 불확실하다. 발을 살살 휘져으면, 꾸준하지만 매우 조금씩밖에 못 내려갈 것이다. (이는 아주 작은 스텝 크기에 비견된다.) 반대로, 욕심껏 빨리 내려가려고 크고 과감하게 발을 내딛을 수도 있는데, 항상 뜻대로 되지는 않을지 모른다. 위의 제시된 코드에서와 같이, 어느 수준 이상의 큰  스켑 크기는 오히려 손실값을 증가시킨다.

<div class="fig figleft fighighlight">
  <img src="{{site.baseurl}}/assets/stepsize.jpg">
  <div class="figcaption">
    작은 변화값(step)이 주는 영향을 시각적으로 보여주는 그림. 특정 지검 W에서 시작해서 그라디언트(혹은 거기에 -1을 곱한 값)를 계산한다. 이 그라디언트에 -1을 곱한 방향, 즉 하얀 화살표 방향이 손실함수(loss function)이 가장 빠르게 감소하는 방향이다. 그 방향으로 조금 가는 것은 일관되지만 느리게 최적화를 진행시킨다. 반면에, 그 방향으로 너무 많이 가면, 더 많이 감소시키지만 위험성도 크다. 스텝 크기가 점점 커지면, 결국에는 최소값을 지나쳐서 손실값이 더 커지는 지점까지 가게될 것이다. 스텝 크기(나중에 <b>학습속도</b>라고 부를 것임) 가장 중요한 하이퍼파라미터(hyperparameter)이라서 매우 조심스럽게 결정해야 할 것이다.
  </div>
  <div style="clear:both;"></div>
</div>

**효율성의 문제**. 알다시피, 그라디언트(gradient)를 수치적으로 계산하는 데 드는 비용은 파라미터(parameter/weight)의 수에 따라 선형적으로 늘어난다. 위 예시에서, 총 30,730의 파라미터(parameter/weight)가 있으므로 30,731번 손실함수값을 계산해서 그라디언트(gradient)를 구해 봐야 딱 한 번 업데이트할 수 있다. 요즘 쓰이는 신경망(neural networks)들은 수천만개의 파라미터(parameter/weight)도 우스운데, 그런 경우 이 문제는 매우 심각해진다. 당연하게도, 이 전략은 별로고, 더 좋은게 있다.

<a name='analytic'></a>

#### 미적분을 이용하여 해석적으로 그라디언트(gradient)를 계산하기

수치적으로 계산하는 그라디언트(gradient)는 유한차이(finite difference)를 이용해서 매우 단순하지만, 단점은 근사값이라는 점과 (그라디언트의 진짜 정의는 "h"가 0으로 수렴할 때의 극한값인데, 여기서는 그냥 작은 "h"값을 쓰기 때문에), 계산이 비효율적이라는 것이다. 두번째 방법은 미적분을 이용해서 해석적으로 그라디언트(gradient)를 구하는 것인데, 이는 (근사치가 아닌) 정확한 수식을 이용하기 때문에 계산하기 매우 빠르다. 하지만, 수치적으로 구한 그라디언트(gradient)와는 다르게, 구현하는데 실수하기 쉽다. 그래서, 실제 응용할 때 해석적으로 구한 다음에 수치적으로 구한 것과 비교해보고, 틀린 경우 고치는 게 흔한 일이다. 이 과정을 **그라디언트체크(gradient check)**라고 한다..

SVM 손실함수(loss function)의 예를 들어서 설명해보자.

$$
L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta) \right]
$$

파라미터(parameter/weight)로 이 함수를 미분할 수 있다. 예를 들어, $w_{y_i}$로 미분하면 이렇게 된다:

$$
\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i
$$

여기서 $\mathbb{1}$은 정의함수라고 하는데, 쉽게 말해 괄호 안의 조건이 충족되면 1, 아니면 0인 값을 갖는다. 이렇게 써놓으면 무시무시해보이지만, 실제로 코딩으로 구현할 때는 원하는 차이(마진, margin)을 못 만족시키는, 따라서 손실함수(loss function)의 증가에 일조하는 클래스의 개수를 세고, 이 숫자를 데이터벡터 $x_i$에 곱하면 이게 바로 그라디언트(gradient)이다. 단, 이는 참인 클래스에 해당하는 $W$의 행으로 미분했을 때의 그라디언트이다. $j \neq y_i $인 다른 행에 대한 그라디언트(gradient)는 다음과 같다.

$$
\nabla_{w_j} L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i
$$

일단 그라디언트(gradient) 수식을 구하고 그라디언트(gradient)를 업데이트시키는 것은 간단하다.

<a name='gd'></a>

### 그라디언트 하강 (gradient descent)

이제 손실함수(loss function)의 그라디언트(gradient)를 계산할 줄 알게 됐는데, 그라디언트(gradient)를 계속해서 계산하고 파라미터(weight/parameter)를 Now that we can compute the gradient of the loss function, the procedure of repeatedly evaluating the gradient and then performing a parameter update is called *Gradient Descent*. Its **vanilla** version looks as follows:

~~~python
# 단순한 경사하강(gradient descent)

while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # 파라미터 업데이트(parameter update)
~~~

이 단순한 루프는 모든 신경망(neural network)의 중심에 있는 것이다. 다른 방법으로 (예컨데. LBFGS) 최적화를 할 수 있는 방법이 있긴 하지만, 현재로는 그라디언트 하강 (gradient descent)이 신경망(neural network)의 손실함수(loss function)을 최적화하는 것으로는 가장 많이 쓰인다. 이 강의에서, 이 루프에 이것저것 세세하게 덧붙이기(예를 들어, 업데이트 수식이 정확히 어떻게 되는지 등)는 할 것이다. 하지만, 결과에 만족할 때까지 그라디언트(gradient)를 따라서 움직인다는 기본적인 개념은 안 바뀔 것이다.
bat
**미니배치 그라디언트 하강 (Mini-batch gradient descent (MGD)).** (ILSVRC challenge처럼) 대규모의 응용사례에서, 학습데이터(training data)가 수백만개 주어질 수 있다. 따라서, 파라미터를 한 번 업데이트하려고 학습데이터(training data) 전체를 계산에 사용하는 것은 낭비가 될 수 있다. 이를 극복하기 위해서 흔하게 쓰이는 방법으로는, 학습데이터(training data)의 **배치(batches)**만 이용해서 그라디언트(gradient)를 구하는 것이다. 예를 들어 ConvNets을 쓸 때, 한 번에 120만개 중에 256개짜리 배치만을 이용해서 그라디언트(gradient)를 구하고 파라미터(parameter/weight) 업데이트를 한다.  다음 코드를 보자.

~~~python
# 단순한 미니배치 (minibatch) 그라디언트(gradient) 업데이트

while True:
  data_batch = sample_training_data(data, 256) # 예제 256개짜리 미니배치(mini-batch)
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # 파라미터 업데이트(parameter update) 
~~~

이 방법이 먹히는 이유는 학습데이터들의 예시들이 서로 상관관계가 있기 때문이다. 이것에 대해 알아보기위해, ILSVRC의 120만개 이미지들이 사실은 1천개의 서로 다른 이미지들의 복사본이라는 극단적인 경우를 생각해보자. (즉, 한 클래스 당 하나이고, 이 하나가 1천2백번 복사된 것) 그러면 명백한 것은, 이 1천2백개의 이미지에서의 그라디언트(gradient)값 (역자 주: 이 1천2백개에 해당하는 $i$에 대한 $L_i$값)은 다 똑같다는 점이다. 그렇다면 이 1천2백개씩 똑같은 값들 120만개를 평균내서 손실값(loss)를 구하는 것이나, 서로 다른 1천개의 이미지당 하나씩 1000개의 값을 평균내서 손실값(loss)을 구하는 것이나 똑같다. 실제로는 당연히 중복된 데이터를 주지는 않겠지만, 미니배치(mini-batch)에서만 계산하는 그라디언트(gradient)는 모든 데이터를 써서 구하는 것의 근사값으로 괜찮게 쓰일 수 있을 것이다. 따라서, 미니배치(mini-batch)에서 그라디언트(gradient)를 구해서 더 자주자주 파라미터(parameter/weight)을 업데이트하면 실제로 더 빠른 수렴하게 된다.

이 방법의 극단적인 형태는 미니배치(mini-batch)가 데이터 달랑 한개로 이루어졌을 때이다. 이는 **확률그라디언트하강(Stochastic Gradient Descent (SGD))** (혹은 **온라인** 그라디언트 하강)이라고 불린다. 이건 상대적으로 덜 보편적인데, 그 이유는 우리가 프로그램을 짤 때 계산을 벡터/행렬로 만들어서 하기 때문에, 한 예제에서 100번 계산하는 것보다 100개의 예제에서 1번 계산하는게 더 빠르기 때문이다. SGD가 엄밀한 의미에서는 예제 하나짜리 미니배치(mini-batch)에서 그라디언트(gradient)를 계산하는 것이지만, 많은 사람들이 그냥 MGD를 의미하면서 SGD라고 부르기도 한다. 혹은 드물게나마 배치 그라디언트 하강 (Batch gradient descent, BGD)이라고도 부른다. 미니배치(mini-batch)의 크기도 하이퍼파라미터(hyperparameter)이지만, 이것을 교차검증하는 일은 흔치 않다. 이건 대체로 컴퓨터 메모리 크기의 한계에 따라 결정되거나, 몇몇 특정값 (예를 들어, 32, 64 or 128 같은 것)을 이용한다. 2의 제곱수를 이용하는 이유는 많은 벡터 계산이 2의 제곱수가 입력될 때 더 빠르기 때문이다.

<a name='summary'></a>

### 요약

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/dataflow.jpeg">
  <div class="figcaption">
    정보 흐름 요약. <b>(x,y)</b>라는 고정된 데이터 쌍이 주어져 있다. 처음에는 무작위로 뽑은 파라미터(parameter/weight)값으로 시작해서 바꿔나간다. 왼쪽에서 오른쪽으로 가면서, 스코어함수(score function)가 각 클래스의 점수를 계산하고 그 값이 <b>f</b> 벡터에 저장된다. 손실함수(loss function)는 두 부분으로 나뉘어 있다. 첫째, 데이터 손실(data loss)은 파라미터(parameter/weight)만으로 계산하는 함수이다. 그라디언트 하강(Gradient Descent) 과정에서, 파라미터(parameter/weight)로 미분한 (혹은 원한다면 데이터 값으로 추가로 미분한... ??? 역자 주: 이 괄호안의 내용은 무슨 소린지 모르겠음.) 그라디언트(gradient)를 계산하고, 이것을 이용해서 파라미터(parameter/weight)값을 업데이트한다.
  </div>
</div>


이 섹션에서 다음을 다루었다.

- 손실함수(loss function)가 **고차원의 울퉁불퉁한 지형**이고, 이 지형에서 아래쪽으로 내려가는 것으로 직관적인 설명을 발전시켰다. 이에 대한 비유는 눈가린 등산객이 하산하는 것이었다. 특히, SVM의 손실함수(loss function)가 부분적으로 선형(linear)인 밥공기 모양이라는 것을 확인했따.
- 손실함수(loss function)을 최적화시킨다는 개념을, 아무 데서나 시작해서 더 나아지는 쪽으로 한걸음 한걸음 나은 쪽으로 가서 최적화시킨다는 **반복적으로 개선**의 측면으로 운을 띄워봤고
- 함수의 **그라디언트(gradient)**는 그 함수값이 감소하는 가장 빠른 방향이라는 점을 알아봤고, 이것을 유한차이(finite difference, 즉 미분할 때 *h*의 값이 유한하다는 의미)를 이용하여 단순무식하게 수치적으로 어림잡아 계산하는 방법도 알아보았다.
- 파라미터(parameter/weight)를 업데이트할 때, 한 번에 얼마나 움직여야하는지(혹은 **학습속도**)를 딱 맞게 설정하는 것이 까다로운 문제라는 것도 알아보았다. 이 값이 너무 낮으면 너무 느려지고, 너무 높으면 빨라지지만 위험한 점이 있다. 이 장단점에 대해 다음 섹션에서 자세하게 알아볼 것이다.
- 그라디언트(gradient)를 계산할 때 **수치적**인 방법과 **해석적**인 방법의 장단점을 알아보았다. 수치적인 그라디언트(gradient)는 단순하지만, 근사값이고 비효율적이다. 해석적인 그라디언트(gradient)는 정확하고 빠르지만 손으로 계산해야 되서 실수를 할 수 있다. 따라서 실제 응용에서는 해석적인 그라디언트(gradient)을 쓰고, **그라디언트 체크(gradient check)**라는 수치적인 그라디언트(gradient)와 비교/검증하는 과정을 거친다.
- 반복적으로 루프(loop)를 돌려서 그라디언트(gradient)를 계산하고 파라미터(parameter/weight)를 업데이트하는 **그라디언트 하강 (Gradient Descent)** 알고리즘을 소개했다.

**예고:** 이 섹션에서 핵심은, 손실함수(loss function)를 파라미터(parameter/weight)로 미분하여 그라디언트(gradient)를 계산하는 법(과 그에 대한 직관적인 이해)가 신경망(neural network)를 디자인하고 학습시키고 이해하는데 있어 가장 중요한 기술이라는 점이다. 다음 섹션에서는, 그라디언(gradient)를 해석적으로 구할 때 연쇄법칙을 이용한, **backpropagation**이라고도 불리는 효율적인 방법에 대해 알아보겠다. 이 방법을 쓰면 컨볼루션 신경망 (Convolutional Neural Networks)을 포함한 모든 종류의 신경망(Neural Networks)에서 쓰이는 상대적으로 일반적인 손실함수(loss function)를 효율적으로 최적화시킬 수 있다.
