---
layout: page
permalink: /optimization-1/
---

Table of Contents:

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

1. 원 이미지의 픽셀들을 넣으면 분류 스코어(class score)를 계산해주는 모수화된(parameterized) **스코어함수(score function)** (예를 들어,  선형함수).
2. 학습(training) 데이타에 어떤 특정 모수(parameter/weight)들을 가지고 스코어함수(score function)를 적용시켰을 때, 실제 class와 얼마나 잘 일치하는지에 따라 그 특정 모수(parameter/weight)들의 질을 측정하는 **손실함수(loss function)**. 여러 종류의 손실함수(예를 들어, Softmax/SVM)가 있다.

구체적으로 말하자면, 다음과 같은 형식을 가진 선형함수 $$ f(x_i, W) =  W x_i $$를 스코어함수(score function)로 쓸 때,  앞에서 다룬 바와 같이 SVM은 다음과 같은 수식으로 표현할 수 있다.:

$$
L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + 1) \right] + \alpha R(W)
$$

예시 $x_i$에 대한 예측값이 실제 값(레이블, labels) $$y_i$$과 같도록 설정된 모수(parameter/weight) $$W$$는 손실(loss)값 $$L$$ 또한 매우 낮게 나온다는 것을 알아보았다. 이제 세번째이자 마지막 핵심요소인 **최적화(optimization)**에 대해서 알아보자. 최적화(optimization)는 손실함수(loss function)을 최소화시카는 모수(parameter/weight, $$W$$)들을 찾는 과정을 뜻한다.

**예고:** 이 세 가지 핵심요소가 어떻게 상호작용하는지 이해한 후에는, 첫번째 요소(모수화된 함수)로 다시 돌아가서 선형함수보다 더 복잡한 형태로 확장시켜볼 것이다.  처음엔 신경망(Neural Networks), 다음엔 컨볼루션 신경망(Convolutional Neural Networks). 손실함수(loss function)와 최적화(optimization) 과정은 거의 변화가 없을 것이다..

<a name='vis'></a>

### 손실함수(loss function)의 시각화

이 강의에서 우리가 다루는 손실함수(loss function)들은 대체로 고차원 공간에서 정의된다. 예를 들어, CIFAR-10의 선형분류기(linear classifier)의 경우 모수(parameter/weight) 행렬은 크기가 [10 x 3073]이고 총 30,730개의 모수(parameter/weight)가 있다. 따라서, 시각화하기가 어려운 면이 있다. 하지만, 고차원 공간을 1차원 직선이나 2차원 평면으로 잘라서 보면 약간의 직관을 얻을 수 있다. 예를 들어, 무작위로 모수(parameter/weight) 행렬 $W$을 하나 뽑는다고 가정해보자. (이는 사실 고차원 공간의 한 점인 셈이다.) 이제 이 점을 직선 하나를 따라 이동시키면서 손실함수(loss function)를 기록해보자. 즉, 무작위로 뽑은 방향 $$W_1$$을 잡고, 이 방향을 따라 가면서 손실함수(loss function)를 계산하는데, 구체적으로 말하면 $$L(W + a W_1)$$에 여러 개의 $$a$$ 값(역자 주: 1차원 스칼라)을 넣어 계산해보는 것이다. 이 과정을 통해 우리는 $$a$$ 값을 x축, 손실함수(loss function) 값을 y축에 놓고 간단한 그래프를 그릴 수 있다. 또한 이 비슷한 것을 2차원으로도 할 수 있다. 여러 $$a, b$$값에 따라  $$ L(W + a W_1 + b W_2) $$을 계산하고(역자 주: $$W_2$$ 역시 $$W_1$$과 같은 식으로 뽑은 무작위 방향), $$a, b$$는 각각 x축과 y축에, 손실함수(loss function) 값 색을 이용해 그리면 된다.

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

이 예시들이 1차원이기 때문에, 데이타 $$x_i$$와 모수(parameter/weight) $$w_j$$는 숫자(역자 주: 즉, 스칼라. 따라서 위 수식에서 전치행렬을 뜻하는 $$T$$ 표시는 필요없음)이다. 예를 들어 $$w_0$$ 를 보면, 몇몇 항들은 $$w_0$$의 선형함수이고 각각은 0에서 꺾인다. 이를 다음과 같이 시각화할 수 있다.

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/svmbowl.png">
  <div class="figcaption">
    손실(loss)를 1차원으로 표현한 그림. x축은 모수(parameter/weight) 하나이고, y축은 손실(loss)이다. 손실(loss)는 여러 항들의 합인데, 그 각각은 특정 모수(parameter/weight)값과 무관하거나, 0에 막혀있는 그 모수(parameter/weight)의 선형함수이다. 전체 SVM 손실은 이 모양의 30,730차원 버전이다.
  </div>
</div>

옆길로 새면, 아마도 밥공기 모양을 보고 SVM 손실함수(loss function)이 일종의 [볼록함수](http://en.wikipedia.org/wiki/Convex_function)라고 생각했을 것이다. 이런 형태의 함수를 효율적으로 최소화하는 문제에 대한 엄청난 양의 연구 성과들이 있다. 스탠포드 강좌 중에서도 이 주제를 다룬 것도 있다. ( [볼록함수 최적화](http://stanford.edu/~boyd/cvxbook/) ). 이 스코어함수(score function) $$f$$를 신경망(neural networks)로 확장시키면, 목적함수(역자 주: 손실함수(loss function))은 더이상 볼록함수가 아니게 되고, 위와 같은 시각화를 해봐도 밥공기 모양 대신 울퉁불퉁하고 복잡한 모양이 보일 것이다.

*미분이 불가능한 손실함수(loss functions)*. 기술적인 설명을 덧붙이자면, $$\max(0,-)$$ 함수 때문에 손실함수(loss functionn)에 *꺾임*이 생기는데, 이 때문에 손실함수(loss functions)는 미분이 불가능해진다. 왜냐하면, 그 꺾이는 부분에서 미분 혹은 그라디언트가 존재하지 않기 때문이다. 하지만, [서브그라디언트(subgradient)](http://en.wikipedia.org/wiki/Subderivative)가 존재하고, 대체로 이를 그라디언트(gradient) 대신 이용한다. 앞으로 이 강의에서는 *그라디언트(gradient)*와 *서브그라디언트(subgradient)*를 구분하지 않고 쓸 것이다.

<a name='최적화'></a>

### 최적화

정리하면, 손실함수(loss function)는 모수(parameter/weight) **W** 행렬의 질을 측정한다. 최적화의 목적은 이 손실함수(loss function)을 최소화시키는 **W**을 찾아내는 것이다. 다음 단락부터 손실함수(loss function)을 최적화하는 방법에 대해서 찬찬히 살펴볼 것이다. 이전에 경험이 있는 사람들이 보면 이 섹션은 좀 이상하다고 생각할지 모르겠다. 왜냐하면, 여기서 쓰인 예제 (즉, SVM 손실(loss))가 볼록함수이기 때문이다. 하지만, 우리의 궁극적인 목적은 신경망(neural networks)를 최적화시키는 것이고, 여기에는 볼록함수 최적화를 위해 고안된 방법들이 쉽사리 통히지 않는다.

<a name='opt1'></a>

#### 전략 #1: 첫번째 매우 나쁜 방법: 무작위 탐색 (Random search)

주어진 모수(parameter/weight) **W**이 얼마나 좋은지를 측정하는 것은 매우 간단하기 때문에, 처음 떠오르는 (매우 나쁜) 생각은, 단순히 무작위로 모수(parameter/weight)을 골라서 넣어보고 넣어 본 값들 중 제일 좋은 값을 기록하는 것이다. 그 과정은 다음과 같다.

~~~python
# assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
# assume Y_train are the labels (e.g. 1D array of 50,000)
# assume the function L evaluates the loss function

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

위의 코드에서, 여러 개의 무작위 모수(parameter/weight) **W**를 넣어봤고, 그 중 몇몇은 다른 것들보다 좋았다. 그래서 그 중 제일 좋은 모수(parameter/weight) **W**을 테스트 데이터에 넣어보면 된다.

~~~python
# Assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict == Yte)
# returns 0.1555
~~~

이 방법으로 얻은 최선의 **W**는 정확도 **15.5%**이다. 완전 무작위 찍기가 단 10%의 정확도를 보이므로, 무식한 방법 치고는 그리 나쁜 것은 아니다.

**핵심 아이디어: 반복적 향상**. 물론 이보다 더 좋은 방법들이 있다. 여기서 핵심 아이디어는, 최선의 모수(parameter/weight) **W**을 찾는 것은 매우 어렵거나 때로는 불가능한 문제(특히 복잡한 신경망(neural network) 전체를 구현할 경우)이지만, 어떤 주어진 모수(parameter/weight) **W**을 조금 개선시키는 일은 훨씬 덜 힘들다는 점이다. 다시 말해, 우리의 접근법은 무작위로 뽑은 **W**에서 출발해서 매번 조금씩 개선시키는 것을 반복하는 것이다.

> 우리의 전략은 무작위로 뽑은 모수(parameter/weight)으로부터 시작해서 반복적으로 조금씩 개선시켜 손실(loss)을 낮추는 것이다.

**눈 가리고 하산하는 것에 비유.** 앞으로 도움이 될만한 비유는, 경사진 지형에서 눈가리개를 하고 점점 아래로 내려오는 자기 자신을 생각해보는 것이다. CIFAR-10의 예시에서, 그 언덕들은 (**W**가 3073 x 10 차원이므로) 30,730차원이다. 언덕의 각 지점에서의 고도가 손실함수(loss function)의 손실값(loss)의 역할을 한다.

<a name='opt2'></a>

#### 전략 #2: 무작위 국소 탐색 (Random Local Search)

처음 떠오르는 전략은, 시작점에서 무작위로 방향을 정해서 발을 살짝 뻗어서 더듬어보고 그게 내리막길길을 때만 한발짝 내딛는 것이다. 구체적으로 말하면, 임의의 $$W$$에서 시작하고, 또다른 임의의 방향 $$ \delta W $$으로 살짝 움직여본다. 만약에 움직여간 자리($$W + \delta W$$)에서의 손실잢(loss)가 더 낮으면, 거기로 움직이고 다시 탐색을 시작한다. 이 과정을 코드로 짜면 다음과 같다.

~~~python
W = np.random.randn(10, 3073) * 0.001 # generate random starting W
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

이전 섹션에서, 모수(parameter/weight) 공간에서 모수(parameter/weight) 벡터를 향상시키는 (즉, 손실값을 더 낮추는) 뱡향을 찾는 시도를 해봤다. 그런데 사실 좋은 방향을 찾기 위해 방향을 무작위로 탐색할 필요가 없다고 한다. (적어도 반지름이 0으로 수렴하는 아주 좁은 근방에서는) 가장 가파르게 감소한다고 수학적으로 검증된 *최선의* 방향을 구할 수 있고, 이 방향을 따라 모수(parameter/weight) 벡터를 움직이면 된다는 것이다. 이 방향이 손실함수(loss function)의 **그라디언트(gradient)**와 관계있다. 눈 가리고 하산하는 것에 비유할 때, 발 밑 지형을 잘 더듬어보고 가장 가파르다는 느낌을 주는 방향으로 내려가는 것에 비견할 수 있다.

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
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
~~~

이 코드는, 위에 주어진 그라디언트(gradient) 식을 이용해서 모든 차원을 하나씩 돌아가면서 그 방향으로 작은 변화 `h`를 줬을 때, 손실함수(loss function)의 값이 얼마나 변하는지를 구해서, 그 방향의 편미분 값을 계산한다. 변수 `grad`에 전체 그라디언트(gradient) 값이 최종적으로 저장된다.

**실제 고려할 사항**. **h**가 0으로 수렴할 때의 극한값이 그라디언트(gradient)의 수학적으로 정의인데, (이 예제에서 나온 것처럼 1e-5 같이) 작은 값이면 충분하다. 이상적으로, 수치적인 문제를 일으키지 않는 수준에서 가장 작은 값을 쓰면 된다. 덧붙여서, 실제 활용할 때, x를 **양 방향으로 변화를 주어서 구한 수식**이 더 좋은 경우가 많다: $ [f(x+h) - f(x-h)] / 2 h $ . 다음  [위키](http://en.wikipedia.org/wiki/Numerical_differentiation)를 보면 자세한 것을 알 수 있다.

위에서 계산한 함수를 이용하면, 아무 함수의 아무 값에서나 그라디언트(gradient)를 계산할 수 있다. 무작위로 뽑은 모수(parameter/weight)값에서 CIFAR-10의 손실함수(loss function)의 그라디언트를 구해본다.:

~~~python

# to use the generic code above we want a function that takes a single argument
# (the weights in our case) so we close over X_train and Y_train
def CIFAR10_loss_fun(W):
  return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001 # random weight vector
df = eval_numerical_gradient(CIFAR10_loss_fun, W) # get the gradient
~~~

그라디언트(gradient)는 각 차원에서 CIFAR-10의 손실함수(loss function)의 기울기를 알려주는데, 그걸 이용해서 모수(parameter/weight)를 업데이트한다.

~~~python
loss_original = CIFAR10_loss_fun(W) # the original loss
print 'original loss: %f' % (loss_original, )

# lets see the effect of multiple step sizes
for step_size_log in [-10, -9, -8, -7, -6, -5,-4,-3,-2,-1]:
  step_size = 10 ** step_size_log
  W_new = W - step_size * df # new position in the weight space
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

**Update in negative gradient direction**. 위 코드에서, 새로운 모수 `W_new`로 업데이트할 때, 그라디언트(gradient) `df`의 반대방향으로 움직인 것을 주목하자. 왜냐하면 우리가 원하는 것은 손실함수(loss function)의 증가가 아니라 감소하는 것이기 때문이다.

**작은 변화값 `h` (step)의 크기가 미치는 영향**. 그라디언트(gradient)에서 알 수 있는 것은 함수값이 가장 빠르게 증가하는 방향이고, 그 방향으로 대체 얼만큼을 가야하는지는 알려주지 않는다. 강의 뒤에서 다루게 되겠지만, 얼만큼 가야하는지(*학습 속도*라고도 함), 즉, `h`값의 크기는 신경망(neural network)를 학습시킬 때 있어 가장 중요한 (그래서 결정하기 까다로운) 하이퍼파라미터(hyperparameter)가 될 것이다. 눈 가리고 하산하는 비유에서, 우리는 우리 발 밑으로 어느 방향이 가장 가파른지 느끼지만, 얼마나 발을 뻗어야할 지는 불확실하다. 발을 살살 휘져으면, 꾸준하지만 매우 조금씩밖에 못 내려갈 것이다. (이는 아주 작은 `h`값에 비견된다.) 반대로, 욕심껏 빨리 내려가려고 크고 과감하게 발을 내딛을 수도 있는데, 항상 뜻대로 되지는 않을지 모른다. 위의 제시된 코드에서와 같이, 어느 수준 이상의 큰 `h`값은 오히려 손실값을 증가시킨다.

<div class="fig figleft fighighlight">
  <img src="{{site.baseurl}}/assets/stepsize.jpg">
  <div class="figcaption">
    Visualizing the effect of step size. We start at some particular spot W and evaluate the gradient (or rather its negative - the white arrow) which tells us the direction of the steepest decrease in the loss function. Small steps are likely to lead to consistent but slow progress. Large steps can lead to better progress but are more risky. Note that eventually, for a large step size we will overshoot and make the loss worse. The step size (or as we will later call it - the <b>learning rate</b>) will become one of the most important hyperparameters that we will have to carefully tune.
  </div>
  <div style="clear:both;"></div>
</div>

**효율성의 문제**. 알다시피, 그라디언트(gradient)를 수치적으로 계산하는 데 드는 비용은 모수(parameter/weight)의 수에 따라 선형적으로 늘어난다. 위 예시에서, 총 30,730의 모수(parameter/weight)가 있으므로 30,731번 손실함수값을 계산해서 그라디언트(gradient)를 구해 봐야 딱 한 번 업데이트할 수 있다. 요즘 쓰이는 신경망(neural networks)들은 수천만개의 모수(parameter/weight)도 우스운데, 그런 경우 이 문제는 매우 심각해진다. 당연하게도, 이 전략은 별로고, 더 좋은게 있다.

<a name='analytic'></a>

#### Computing the gradient analytically with Calculus

The numerical gradient is very simple to compute using the finite difference approximation, but the downside is that it is approximate (since we have to pick a small value of *h*, while the true gradient is defined as the limit as *h* goes to zero), and that it is very computationally expensive to compute. The second way to compute the gradient is analytically using Calculus, which allows us to derive a direct formula for the gradient (no approximations) that is also very fast to compute. However, unlike the numerical gradient it can be more error prone to implement, which is why in practice it is very common to compute the analytic gradient and compare it to the numerical gradient to check the correctnes of your implementation. This is called a **gradient check**.

Lets use the example of the SVM loss function for a single datapoint:

$$
L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta) \right]
$$

We can differentiate the function with respect to the weights. For example, taking the gradient with respect to $w_{y_i}$ we obtain:

$$
\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i
$$

where $\mathbb{1}$ is the indicator function that is one if the condition inside is true or zero otherwise. While the expression may look scary when it is written out, when you're implementing this in code you'd simply count the number of classes that didn't meet the desired margin (and hence contributed to the loss function) and then the data vector $x_i$ scaled by this number is the gradient. Notice that this is the gradient only with respect to the row of $W$ that corresponds to the correct class. For the other rows where $j \neq y_i $ the gradient is:

$$
\nabla_{w_j} L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i
$$

Once you derive the expression for the gradient it is straight-forward to implement the expressions and use them to perform the gradient update.

<a name='gd'></a>

### Gradient Descent

Now that we can compute the gradient of the loss function, the procedure of repeatedly evaluating the gradient and then performing a parameter update is called *Gradient Descent*. Its **vanilla** version looks as follows:

~~~python
# Vanilla Gradient Descent

while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update
~~~

This simple loop is at the core of all Neural Network libraries. There are other ways of performing the optimization (e.g. LBFGS), but Gradient Descent is currently by far the most common and established way of optimizing Neural Network loss functions. Throughout the class we will put some bells and whistles on the details of this loop (e.g. the exact details of the update equation), but the core idea of following the gradient until we're happy with the results will remain the same.

**Mini-batch gradient descent.** In large-scale applications (such as the ILSVRC challenge), the training data can have on order of millions of examples. Hence, it seems wasteful to compute the full loss function over the entire training set in order to perform only a single parameter update. A very common approach to addressing this challenge is to compute the gradient over **batches** of the training data. For example, in current state of the art ConvNets, a typical batch contains 256 examples from the entire training set of 1.2 million. This batch is then used to perform a parameter update:

~~~python
# Vanilla Minibatch Gradient Descent

while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
~~~

The reason this works well is that the examples in the training data are correlated. To see this, consider the extreme case where all 1.2 million images in ILSVRC are in fact made up of exact duplicates of only 1000 unique images (one for each class, or in other words 1200 identical copies of each image). Then it is clear that the gradients we would compute for all 1200 identical copies would all be the same, and when we average the data loss over all 1.2 million images we would get the exact same loss as if we only evaluated on a small subset of 1000. In practice of course, the dataset would not contain duplicate images, the gradient from a mini-batch is a good approximation of the gradient of the full objective. Therefore, much faster convergence can be achieved in practice by evaluating the mini-batch gradients to perform more frequent parameter updates.

The extreme case of this is a setting where the mini-batch contains only a single example. This process is called **Stochastic Gradient Descent (SGD)** (or also sometimes **on-line** gradient descent). This is relatively less common to see because in practice due to vectorized code optimizations it can be computationally much more efficient to evaluate the gradient for 100 examples, than the gradient for one example 100 times. Even though SGD technically refers to using a single example at a time to evaluate the gradient, you will hear people use the term SGD even when referring to mini-batch gradient descent (i.e. mentions of MGD for "Minibatch Gradient Descent", or BGD for "Batch gradient descent" are rare to see), where it is usually assumed that mini-batches are used. The size of the mini-batch is a hyperparameter but it is not very common to cross-validate it. It is usually based on memory constraints (if any), or set to some value, e.g. 32, 64 or 128. We use powers of 2 in practice because many vectorized operation implementations work faster when their inputs are sized in powers of 2.

<a name='summary'></a>

### Summary

<div class="fig figcenter fighighlight">
  <img src="{{site.baseurl}}/assets/dataflow.jpeg">
  <div class="figcaption">
    Summary of the information flow. The dataset of pairs of <b>(x,y)</b> is given and fixed. The weights start out as random numbers and can change. During the forward pass the score function computes class scores, stored in vector <b>f</b>. The loss function contains two components: The data loss computes the compatibility between the scores <b>f</b> and the labels <b>y</b>. The regularization loss is only a function of the weights. During Gradient Descent, we compute the gradient on the weights (and optionally on data if we wish) and use them to perform a parameter update during Gradient Descent.
  </div>
</div>


In this section,

- We developed the intuition of the loss function as a **high-dimensional optimization landscape** in which we are trying to reach the bottom. The working analogy we developed was that of a blindfolded hiker who wishes to reach the bottom. In particular, we saw that the SVM cost function is piece-wise linear and bowl-shaped.
- We motivated the idea of optimizing the loss function with
**iterative refinement**, where we start with a random set of weights and refine them step by step until the loss is minimized.
- We saw that the **gradient** of a function gives the steepest ascent direction and we discussed a simple but inefficient way of computing it numerically using the finite difference approximation (the finite difference being the value of *h* used in computing the numerical gradient).
- We saw that the parameter update requires a tricky setting of the **step size** (or the **learning rate**) that must be set just right: if it is too low the progress is steady but slow. If it is too high the progress can be faster, but more risky. We will explore this tradeoff in much more detail in future sections.
- We discussed the tradeoffs between computing the **numerical** and **analytic** gradient. The numerical gradient is simple but it is approximate and expensive to compute. The analytic gradient is exact, fast to compute but more error-prone since it requires the derivation of the gradient with math. Hence, in practice we always use the analytic gradient and then perform a **gradient check**, in which its implementation is compared to the numerical gradient.
- We introduced the **Gradient Descent** algorithm which iteratively computes the gradient and performs a parameter update in loop.

**Coming up:** The core takeaway from this section is that the ability to compute the gradient of a loss function with respect to its weights (and have some intuitive understanding of it) is the most important skill needed to design, train and understand neural networks. In the next section we will develop proficiency in computing the gradient analytically using the chain rule, otherwise also refered to as **backpropagation**. This will allow us to efficiently optimize relatively arbitrary loss functions that express all kinds of Neural Networks, including Convolutional Neural Networks.
