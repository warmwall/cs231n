---
layout: page
permalink: /optimization-2/
---

Table of Contents:

- [소개(Introduction)](#intro)
- [그라디언트(Gradient)에 대한 간단한 표현과 이해](#grad)
- [복합 표현식(Compound Expression), 연쇄 법칙(Chain rule), Backpropagation](#backprop)
- [Backpropation에 대한 직관적인 이해](#intuitive)
- [모듈성 : 시그모이드(Sigmoid) 예제](#sigmoid)
- [Backprop 실제: 단계별 계산](#staged)
- [역박향 흐름의 패턴](#patters)
- [벡터 기반의 그라디언트(Gradient) 계산)](#mat)
- [요약](#summary)

<a name='intro'></a>

### Introduction

**Motivation**. 이번 섹션에서 우리는 **Backpropagation**에 대한 직관적인 이해를 바탕으로 전문지식을 더 키우고자 한다. Backpropagation은 네트워크 전체에 대해 반복적인 **연쇄 법칙(Chain rule)**을 적용하여 그라디언트(Gradient)를 계산하는 방법 중 하나이다. Backpropagation 과정과 세부 요소들에 대한 이해는 여러분에게 있어서 신경망을 효과적으로 개발하고, 디자인하고 디버그하는 데 중요하다고 볼 수 있다. 

**Problem statement**. 이번 섹션에서 공부할 핵심 문제는 다음과 같다 : 주어진 함수 $$f(x)$$ 가 있고, $$x$$ 는 입력 값으로 이루어진 벡터이고, 주어진 입력 $$x$$에 대해서 함수 $$f$$의 그라디언트를 계산하고자 한다. (i.e. $$\nabla f(x)$$ ).

**Motivation**. 우리가 이 문제에 관심을 기울이는 이유에 대해 신경망 관점에서 좀더 구체적으로 살펴 보자. $$f$$는 손실 함수 ( $$L$$ ) 에 해당하고 입력 값 $$x$$ 는 학습 데이터(Training data)와 신경망의 Weight라고 볼 수 있다. 예를 들면, 손실 함수는 SVM Loss 함수가 될 수 있고, 입력 값은 학습 데이터 $$(x_i,y_i), i=1 \ldots N$$ 와 Weight, Bias $$W,b$$ 으로 볼 수 있다. 여기서 학습데이터는 미리 주어져서 고정 되어있는 값으로 볼 수 있고 (보통의 기계 학습에서 그러하듯..), Weight는 신경망의 학습을 위해 실제로 컨트롤 하는 값이다. 따라서 입력 값 $$x_i$$ 에 대한 그라디언트 계산이 쉬울지라도, 실제로는 파라미터(Parameter) 값에 대한 그라디언트를 일반적으로 계산하고, 그라디언트 값을 활용하여 파라미터를 업데이트 할 수 있다. 하지만, 신경망이 어떻게 작동하는지 해석하고, 시각화 하는 부분에서 입력 값 $x_i$에 대한 그라디언트도 유용하게 활용 될 수 있는데, 이 부분은 본 강의의 뒷부분에 다룰 예정이다. 


여러분이 이미 연쇄 법칙을 통해 그라디언트를 도출하는데 익숙하더라도 이 섹션을 간략히 훑어보기를 권장한다. 왜냐하면 이 섹션에서는 다른데서는 보기 힘든 Backpropagation에 대한 실제 숫자를 활용한 역방향 흐름(Backward Flow)에 대해 설명을 할 것이고, 이를 통해 여러분이 얻게 될 통찰력은 이번 강의 전체에 있어 도움이 될 것이라 생각하기 때문이다.

<a name='grad'></a>

### 그라디언트(Gradient)에 대한 간단한 표현과 이해

복잡한 모델에 대한 수식등을 만들기에 앞서 간단하게 시작을 해보자. x와 y 두 숫자의 곱을 계산하는 간단한 함수 f를 정의하자. $$f(x,y) = x y$$. 각각의 입력 변수에 대한 편미분은 간단한 수학으로 아래와 같이 구해 진다. : 

$$
f(x,y) = x y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = y \hspace{0.5in} \frac{\partial f}{\partial y} = x 
$$

**Interpretation**. 미분이 여러분에게 시사하는 바를 명심하자 : 미분은 입력 변수 부근의 아주 작은(0에 매우 가까운) 변화에 대한 해당 함수 값의 변화량이다. : 

$$
\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}
$$

위에 수식을 기술적인 관점에서 보면, 왼쪽에 있는 분수 기호(가로바)는 오른쪽 분수 기호와 달리 나누기를 뜻하지는 않는다. 대신 연산자 $$  \frac{d}{dx} $$ 가 함수 $$f$$에 적용 되어 미분 된 함수를 의미 하는 것이다. 위의 수식을 이해하는 가장 좋은 방법은 $$h$$가 매우 작으면 함수 $$f$$는 직선으로 근사(Approximated) 될 수 있고, 미분 값은 그 직선의 기울기를 뜻한다. 다시말해, 만약 $$x = 4, y = -3$$ 이면 $$f(x,y) = -12$$ 가 되고, $$x$$에 대한 편미분 값은 $$x$$ $$\frac{\partial f}{\partial x} = -3$$ 으로 얻어진다. 이말은 즉슨, 우리가 x를 아주 조금 증가 시키면 전체 함수 값은 3배로 작아진다는 의미이다. (미분 값이 음수이므로). 이 것은 위의 수식을 재구성하면 이와 같이 간단히 보여 줄 수 있다 ( $$ f(x + h) = f(x) + h \frac{df(x)}{dx} $$ ). 비슷하게, $$\frac{\partial f}{\partial y} = 4$$, 이므로, $$y$$ 값을 아주 작은 $$h$$ 만큼 증가 시킨다면 $$4h$$ 만큼 전체 함수 값은 증가하게 될 것이다. (이번에는 미분 값이 양수)

> 미분은 각 변수가 해당 값에서 전체 함수(Expression)의 결과 값에 영향을 미치는 민감도와 같은 개념이다.

앞서 말했듯이, 그라디언트 $$\nabla f$$는 편미분 값들의 벡터이다. 따라서 수식으로 표현하면 다음과 같다: $$\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x]$$, 그라디언트가 기술적으로 벡터일지라도 심플한 표현을 위해 *"X에 대한 편미분"* 이라는 정확한 표현 대신 *"X에 대한 그라디언트"* 와 같은 표현을 종종 쓰게 될 예정이다. 

다음과 같은 수식에 대해서도 미분값(그라디언트)을 한번 구해보자:

$$
f(x,y) = x + y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = 1 \hspace{0.5in} \frac{\partial f}{\partial y} = 1
$$

위의 수식에서 볼 수 있듯이, $$x,y$$에 대한 미분은 $$x,y$$ 값에 관계 없이 1이다. 당연히, $$x,y$$ 값이 증가하면 $$f$$가 증가하기 때문이다. 그리고 $$f$$ 값의 증가율 또한 $$x,y$$ 값에 관계 없이 일정하다 (앞서 살펴본 곱셈의 경우와 다름). 마지막으로 살펴볼 함수는 우리가 수업에서 자주 다루는 *Max* 함수 이다 :

$$
f(x,y) = \max(x, y) \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = \mathbb{1}(x >= y) \hspace{0.5in} \frac{\partial f}{\partial y} = \mathbb{1}(y >= x)
$$

입력 값이 더 큰 값에 대한 (서브)그라디언트는 1이고, 다른 입력 값의 그라디언트는 0이 된다. 직관적으로 보면, $$x = 4,y = 2$$ 인 경우 max 값은 4 이고, 이 함수는 현재의 $$y$$ 값에 영향을 받지 않는다. 바꾸어말하면, $$y$$값을 아주 작은 값인 $$h$$ 만큼 증가시키더라도 이 함수의 결과 값은 4로 유지된다. 따라서 그라디언트는 0이다 (y값의 영향이 없다). 물론 $$y$$값을 매우 크게 증가 시킨다면 (예를 들면 2이상) 함수 $$f$$ 값은 바뀌겠지만, 미분은 이런 큰 변화 값과는 관련이 없다. 미분이라는 것이 본래 그 정의에도 있듯($$\lim_{h \rightarrow 0}$$) 아주 작은 입력 값 변화에 대해서 의미를 갖는 값이기 때문이다.

<a name='backprop'></a>

### 연쇄 법칙(Chain rule)을 이용한 복합 표현식

이제 $f(x,y,z) = (x + y) z$ 같은 다수의 복합 함수(composed functions)를 수반하는 더 복잡한 표현식을 고려해보자. 이 표현식은 여전히 바로 미분하기에 충분히 간단하지만, 우리는 이 식에 특별한 접근법을 적용할 것이다. 이는 backpropagation 뒤에 있는 직관을 이해하는데 도움이 될 것이다. 특히 이 식이 두 개의 표현식 $q = x + y$와 $f = q z$ 으로 분해될 수 있음에 주목하자. 게다가 이전 섹션에서 본 것처럼 우리는 두 식에 대한 미분값을 어떻게 따로따로 계산할지 알고 있다. $f$ 는 단지 $q$와 $z$의 곱이다. 따라서 $\frac{\partial f}{\partial q} = z, \frac{\partial f}{\partial z} = q$, 그리고 $q$는 $x$와 $y$의 합이므로 $\frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1$이다. 하지만, 중간 결과값인 $q$에 대한 그라디언트($\frac{\partial f}{\partial q}$)를 신경쓸 필요가 없다. 대신 궁극적으로 입력 $x,y,z$에 대한 $f$의 그라디언트에 관심이 있다. **연쇄 법칙**은 이러한 그라디언트 표현식들을 함께 연결시키는 적절한 방법이 곱하는 것이라는 것을 보여준다. 예를 들면, $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x} $와 같이 표현할 수 있다. 실제로 이는 단순히 두 그라디언트를 담고 있는 두 수의 곱셈이다. 하나의 예를 통해 확인 해보자.

~~~python
# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
# now backprop through q = x + y
dfdx = 1.0 * dfdq # dq/dx = 1. And the multiplication here is the chain rule!
dfdy = 1.0 * dfdq # dq/dy = 1
~~~

결국 `[dfdx,dfdy,dfdz]` 변수들로 그라디언트가 표현되는데, 이는 `f`에 대한 변수 `x,y,z`의 민감도(sensitivity)를 보여준다. 이는 backpropagation의 가장 간단한 예이다. 더 나아가서 보다 간결한 표기법을 사용해서 `df` 파트를 계속 쓸 필요가 없도록 하고 싶을 것이다. 예를 들어 `dfdq` 대신에 단순히 `dq`를 쓰고 항상 그라디언트가 최종 출력에 관한 것이라 가정하는 것이다.

또한 이런 계산은 회로도를 가지고 다음과 같이 멋지게 시각화할 수 있다:

<div class="fig figleft fighighlight">
<svg width="420" height="220"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="40" y1="30" x2="110" y2="30" stroke="black" stroke-width="1"></line><text x="45" y="24" font-size="16" fill="green">-2</text><text x="45" y="47" font-size="16" fill="red">-4</text><text x="35" y="24" font-size="16" text-anchor="end" fill="black">x</text><line x1="40" y1="100" x2="110" y2="100" stroke="black" stroke-width="1"></line><text x="45" y="94" font-size="16" fill="green">5</text><text x="45" y="117" font-size="16" fill="red">-4</text><text x="35" y="94" font-size="16" text-anchor="end" fill="black">y</text><line x1="40" y1="170" x2="110" y2="170" stroke="black" stroke-width="1"></line><text x="45" y="164" font-size="16" fill="green">-4</text><text x="45" y="187" font-size="16" fill="red">3</text><text x="35" y="164" font-size="16" text-anchor="end" fill="black">z</text><line x1="210" y1="65" x2="280" y2="65" stroke="black" stroke-width="1"></line><text x="215" y="59" font-size="16" fill="green">3</text><text x="215" y="82" font-size="16" fill="red">-4</text><text x="205" y="59" font-size="16" text-anchor="end" fill="black">q</text><circle cx="170" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="170" y="70" font-size="20" fill="black" text-anchor="middle">+</text><line x1="110" y1="30" x2="150" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="100" x2="150" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="190" y1="65" x2="210" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="380" y1="117" x2="450" y2="117" stroke="black" stroke-width="1"></line><text x="385" y="111" font-size="16" fill="green">-12</text><text x="385" y="134" font-size="16" fill="red">1</text><text x="375" y="111" font-size="16" text-anchor="end" fill="black">f</text><circle cx="340" cy="117" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="340" y="127" font-size="20" fill="black" text-anchor="middle">*</text><line x1="280" y1="65" x2="320" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="170" x2="320" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="360" y1="117" x2="380" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></svg>

<div class="figcaption">
  좌측에 실수 값으로 표현되는 <i>"회로"</i>는 이 계산에 대한 시각 표현을 보여준다. <b>전방 전달(forward pass)</b>은 입력부터 출력까지 값을 계산한다 (녹색으로 표시). 그리고 나서 <b>후방 전달(backward pass)</b>은 backpropagation을 수행하는데, 이는 끝에서 시작해서 반복적으로 연쇄 법칙을 적용해 회로 입력에 대한 모든 길에서 그라디언트 값(적색으로 표시)을 계산한다. 그라디언트 값은 회로를 통해 거꾸로 흐르는 것으로 볼 수 있다.
</div>
<div style="clear:both;"></div>
</div>

<a name='intuitive'></a>
### Backpropagation에 대한 직관적 이해

backpropagation이 굉장히 지역적인(local) 프로세스임에 주목하자. 회로도 내의 모든 게이트(gate) 몇개의 입력을 받아드리고 곧 바로 두 가지를 계산할 수 있다: 1. 게이트의 출력 값, 2. 게이트 출력에 대한 입력들의 *지역적* 그라디언트 값. 여기서 게이트들이 포함된 전체 회로의 세세한 부분을 모르더라도 완전히 독립적으로 값들을 계산할 수 있음을 주목하라. 하지만, 일단 전방 전달이 끝나면 backpropagation 과정에서 게이트는 결국 전체 회로의 마지막 출력에 대한 게이트 출력의 그라디언트 값에 관해 학습할 것이다. 연쇄 법칙을 통해 게이트는 이 그라디언트 값을 받아들여 모든 입력에 대해서 계산한 게이트의 모든 그라디언트 값에 곱한다.

> 연쇄 법칙 덕분에 이러한 각 입력에 대한 추가 곱셈은 전체 신경망과 같은 복잡한 회로에서 상대적으로 쓸모 없는 개개의 게이트를 중요하지 않은 것으로 바꿀 수 있다.

다시 위 예를 통해 이것이 어떻게 동작하는지에 대한 직관을 얻자. 덧셈 게이트는 입력 [-2, 5]를 받아 3을 출력한다. 이 게이트는 덧셈 연산을 하고 있기 때문에 두 입력에 대한 게이트의 지역적 그라디언트 값은 +1이 된다. 회로의 나머지 부분을 통해 최종 출력 값으로 -12가 나온다. 연쇄 법칙이 회로를 역으로 가로질러 반복적으로 적용되는 후방 전달 과정 동안, (곱셈 게이트의 입력인) 덧셈 게이트는 출력 값에 대한 그라디언트 값이 -4였다는 것을 학습한다. 만약 회로가 높은 값을 출력하기를 원하는 것으로 의인화하면 (이는 직관에 도움이 될 수 있다), 이 회로가 덧셈 게이트의 출력 값이 4의 *힘*으로 낮아지길 (음의 부호이기 때문) "원하는" 것으로 볼 수 있다. 반복을 지속하고 그라디언트 값을 연결하기 위해 덧셈 게이트는 이 그라디언트 값을 받아들이고 이를 모든 입력들에 대한 지역적 그라디언트 값에 곱한다 (**x**와 **y**에 대한 그라디언트 값이 1 * -4 = -4가 되도록). 다음의 원하는 효과가 있다는 사실에 주목하자. 만약 **x,y**가 (음의 그라디언트 값에 대한 반응으로) 감소한다면, 이 덧셈 게이트의 출력은 감소할 것이고 이는 다시 곱셈 게이트의 출력이 증가하도록 만들 것이다. 

따라서 backpropagation은 보다 큰 최종 출력 값을 얻도록 게이트들이 자신들의 출력이 (얼마나 강하게) 증가하길 원하는지 또는 감소하길 원하는지 서로 소통하는 것으로 간주할 수 있다.

<a name='sigmoid'></a>
### 모듈성: 시그모이드(Sigmoid) 예제

위에서 본 게이트들은 상대적으로 임의로 선택된 것이다. 어떤 종류의 함수도 미분가능하다면 게이트로서 역할을 할 수 있다. 필요한 경우 여러 개의 게이트를 그룹지어서 하나의 게이트로 만들거나, 하나의 함수를 여러 개의 게이트로 분해할 수도 있다. 이러한 요점을 보여주는 다른 표현식을 살펴보자:

$$
f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}}
$$

나중에 다른 수업에서 보겠지만, 이 표현식은 *시그모이드 활성* 함수를 사용하는 2차원 뉴런(입력 **x**와 가중치 **w**를 갖는)을 나타낸다. 그러나 지금은 이를 매우 단순하게 *w,x*를 입력으로 받아 하나의 단일 숫자를 출력하는 하나의 함수정도로 생각하자. 이 함수는 여러개의 게이트로 구성된다. 위에서 이미 설명한 게이트들(덧셈, 곱셈, 최대)에 더해 네 종류의 게이트가 더 있다:

$$
f(x) = \frac{1}{x} 
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = -1/x^2 
\\\\
f_c(x) = c + x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = 1 
\\\\
f(x) = e^x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = e^x
\\\\
f_a(x) = ax
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = a
$$

여기서 $f_c, f_a$는 각각 입력을 상수 $c$만큼 이동시키고, 상수 $a$만큼 크기를 조정하는 함수이다. 이 함수들은 덧셈과 곰셈의 기술적으로 특별한 경우에 해당하지만, 여기서는 상수 $c,a$에 대한 그라디언트가 필요한 것이기에 (새로운) 단일 게이트로써 소개하고자 한다. 그러면 전체 회로는 다음과 같이 나타난다.

<div class="fig figleft fighighlight">
<svg width="799" height="306"><g transform="scale(0.8)"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="50" y1="30" x2="90" y2="30" stroke="black" stroke-width="1"></line><text x="55" y="24" font-size="16" fill="green">2.00</text><text x="55" y="47" font-size="16" fill="red">-0.20</text><text x="45" y="24" font-size="16" text-anchor="end" fill="black">w0</text><line x1="50" y1="100" x2="90" y2="100" stroke="black" stroke-width="1"></line><text x="55" y="94" font-size="16" fill="green">-1.00</text><text x="55" y="117" font-size="16" fill="red">0.39</text><text x="45" y="94" font-size="16" text-anchor="end" fill="black">x0</text><line x1="50" y1="170" x2="90" y2="170" stroke="black" stroke-width="1"></line><text x="55" y="164" font-size="16" fill="green">-3.00</text><text x="55" y="187" font-size="16" fill="red">-0.39</text><text x="45" y="164" font-size="16" text-anchor="end" fill="black">w1</text><line x1="50" y1="240" x2="90" y2="240" stroke="black" stroke-width="1"></line><text x="55" y="234" font-size="16" fill="green">-2.00</text><text x="55" y="257" font-size="16" fill="red">-0.59</text><text x="45" y="234" font-size="16" text-anchor="end" fill="black">x1</text><line x1="50" y1="310" x2="90" y2="310" stroke="black" stroke-width="1"></line><text x="55" y="304" font-size="16" fill="green">-3.00</text><text x="55" y="327" font-size="16" fill="red">0.20</text><text x="45" y="304" font-size="16" text-anchor="end" fill="black">w2</text><line x1="170" y1="65" x2="210" y2="65" stroke="black" stroke-width="1"></line><text x="175" y="59" font-size="16" fill="green">-2.00</text><text x="175" y="82" font-size="16" fill="red">0.20</text><circle cx="130" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="75" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="30" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="100" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="65" x2="170" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="170" y1="205" x2="210" y2="205" stroke="black" stroke-width="1"></line><text x="175" y="199" font-size="16" fill="green">6.00</text><text x="175" y="222" font-size="16" fill="red">0.20</text><circle cx="130" cy="205" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="215" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="170" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="240" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="205" x2="170" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="290" y1="135" x2="330" y2="135" stroke="black" stroke-width="1"></line><text x="295" y="129" font-size="16" fill="green">4.00</text><text x="295" y="152" font-size="16" fill="red">0.20</text><circle cx="250" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="250" y="140" font-size="20" fill="black" text-anchor="middle">+</text><line x1="210" y1="65" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="210" y1="205" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="270" y1="135" x2="290" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="410" y1="222" x2="450" y2="222" stroke="black" stroke-width="1"></line><text x="415" y="216" font-size="16" fill="green">1.00</text><text x="415" y="239" font-size="16" fill="red">0.20</text><circle cx="370" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="370" y="227" font-size="20" fill="black" text-anchor="middle">+</text><line x1="330" y1="135" x2="350" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="310" x2="350" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="390" y1="222" x2="410" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="530" y1="222" x2="570" y2="222" stroke="black" stroke-width="1"></line><text x="535" y="216" font-size="16" fill="green">-1.00</text><text x="535" y="239" font-size="16" fill="red">-0.20</text><circle cx="490" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="490" y="227" font-size="20" fill="black" text-anchor="middle">*-1</text><line x1="450" y1="222" x2="470" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="510" y1="222" x2="530" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="650" y1="222" x2="690" y2="222" stroke="black" stroke-width="1"></line><text x="655" y="216" font-size="16" fill="green">0.37</text><text x="655" y="239" font-size="16" fill="red">-0.53</text><circle cx="610" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="610" y="227" font-size="20" fill="black" text-anchor="middle">exp</text><line x1="570" y1="222" x2="590" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="630" y1="222" x2="650" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="770" y1="222" x2="810" y2="222" stroke="black" stroke-width="1"></line><text x="775" y="216" font-size="16" fill="green">1.37</text><text x="775" y="239" font-size="16" fill="red">-0.53</text><circle cx="730" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="730" y="227" font-size="20" fill="black" text-anchor="middle">+1</text><line x1="690" y1="222" x2="710" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="750" y1="222" x2="770" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="890" y1="222" x2="930" y2="222" stroke="black" stroke-width="1"></line><text x="895" y="216" font-size="16" fill="green">0.73</text><text x="895" y="239" font-size="16" fill="red">1.00</text><circle cx="850" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="850" y="227" font-size="20" fill="black" text-anchor="middle">1/x</text><line x1="810" y1="222" x2="830" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="870" y1="222" x2="890" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></g></svg>
<div class="figcaption">
  시그모이드 활성 함수를 갖는 2차원 뉴런에 대한 예시 회로. 입력은 [x0,x1]이고 뉴런의 (학습 가능한) 파라미터 값들은 [w0,w1,w2]이다. 나중에 보겠지만, 뉴런은 입력을 가지고 내적을 계산하고 이 입력의 활성 함수 출력 값은 0부터 1사이의 범위에 들어가도록 시그모이드 함수에 의해 압착(squash)이 된다.
</div>
<div style="clear:both;"></div>
</div>

위 예제에서 **w,x** 사이의 내적의 결과로 동작하는 함수 적용(function applications)의 긴 체인을 보았다. 이러한 연산을 제공하는 함수를 *시그모이드 함수(sigmoid function)* $\sigma(x)$ 라고 한다. 만약 (분자에 1을 더하고 다시 빼는 재미있지만 까다로운 과정을 거친 후에)미분을 한다면 입력에 대한 시그모이드 함수의 미분값은 단순화할 수 있는 것으로 알려져 있다.

$$
\sigma(x) = \frac{1}{1+e^{-x}} \\\\
\rightarrow \hspace{0.3in} \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left( \frac{1 + e^{-x} - 1}{1 + e^{-x}} \right) \left( \frac{1}{1+e^{-x}} \right) 
= \left( 1 - \sigma(x) \right) \sigma(x)
$$

보이는 것처럼 그라디언트는 단순화되면서 놀라울만큼 간단해진다.예를 들어 시그모이드 표현은 전방 전달(forward pass) 과정에서 입력 1.0을 받아 출력 0.73을 계산한다. 단일의 단순하고 효율적인 표현식을 이용해 (그리고 더 적은 수치적인 문제를 갖고) 계산하는 방식을 제외하고서, 마치 이전에 본 회로가 계산했던 것(위 그림을 보라)과 비슷하게 위의 미분은 *지역(local)* 그라디언트 값이 단순히 (1 - 0.73) * 0.73 ~= 0.2 가 됨을 보여준다. 그러므로 어떤 실제 실용적인 적용에서 그러한 연산들을 단일 게이트로 묶어주는 것은 매우 유용하다고 할 수 있다. 코드에서 이 뉴런에 대한 backprop를 살펴보자:

~~~python
w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit
~~~

**구현 팁(protip): 단계적 backpropagation**. 위 코드에서 볼 수 있듯이, 전방 전달(forward pass)를 쉽게 backprop되는 단계들로 잘게 분해하는 것은 실질적으로 항상 도움이 된다. 예를 들어 우리는 여기서 `w`와 `x` 사이의 내적의 결과를 담는 중간 변수 `dot`를 만들었다. 그리고나서 후방 전달(backward pass) 과정에서 그러한 변수들의 그라디언트 값들을 담은 해당 변수들(예: `ddot` 및 궁극적으로는 `dw, dx`)을 성공적으로 계산한다(역순으로).

이 섹션에서 요점은 어떻게 backpropagation이 수행되는 지와 전방 함수(forward function)의 어느 부분을 게이트로 취급해야할 지에 대한 세부사항은 편의성 문제라는 것이다. 이는 표현식의 어느 부분들이 쉬운 지역 그라디언트를 가지며, 가장 적은 코드의 양과 노력으로 이들을 함께 묶을 수 있는지를 이해하는데 도움이 된다.

<a name='staged'></a>
### 실제 backprop: 단계적 계산

또 다른 예제를 통해 확인해보자. 다음과 같은 형태의 함수가 있다고 가정하자:

$$
f(x,y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}
$$

명확히 말하면, 실제 backpropagation의 좋은 예제라는 사실 외에는 이 함수는 완전히 쓸모가 없으며 따라서 왜 여러분이 이 함수의 그라디언트를 그토록 계산해야 하는지 그 이유도 뚜렷하지 않다. 만약 여러분들이 $x$ 또는 $y$에 관해서 미분을 수행한다면 결국 매우 크고 복잡한 식을 얻게 될 것이다. 하지만, 그라디언트를 계산하는 명확한 함수(explicit function)를 쓸 필요가 없기 때문에 그렇게 미분하는 것은 완전히 불필요한 것으로 알려져있다. 우리는 단지 어떻게 이를 계산하는지만 알면 된다. 다음은 우리가 어떻게 그러한 표현식에 대해 전방 전달(forward pass)을 구조화 하는지를 나타낸 것이다:

~~~python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done!                                 #(8)
~~~

표현식의 마지막에서 전방 전달(forward pass)을 계산했다. 각각이 단순한 표현식들인 다수의 중간 변수들을 포함하는 방식으로 코드를 구조화한 것에 주목하자, 우리는 이미 이 표현식들에 대한 지역 그라디언트 값을 알고 있다. 그러므로, backprop 전달을 계산하는 것은 쉬운 일이다: 전방 전달 과정의 모든 변수들(`sigy, num, sigx, xpy, xpysqr, den, invden`)에 대해 역방향으로 가면서 똑같은 변수들을 볼 것이다, 다만 해당 변수에 대한 회로 출력의 그라디언트를 담는 것을 나타내기 위해 변수명 앞에 `d`를 붙인다. 추가로, backprop에서 모든 단일 조각이 이 표현식에 대한 지역 그라디언트을 계산하고 곱셈 형태로 이 그라디언트 값을 연결하는 과정을 수반할 것이다. 각 행마다 전방 전달 과정에서 어느 부분에 해당하는지 표시한 것이다:

~~~python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
~~~

몇 가지 주의할 점:

**전방 전달 변수들을 저장(cache)하라**. 후방 전달을 계산하기 위해 전방 전달에서 사용한 일부 변수들을 가지고 있는 것은 정말 유용하다. 실제로 여러분은 이 변수들을 저장해서 backpropagation 동안 이용할 수 있도록 코드를 구성하고 싶을 것이다. 이것이 너무 어려운 일이라면, 이 변수들을 다시 계산할 수 있다(물론 비효율적이지만).

**갈래길에서 그라디언트는 더해진다**. 전방 표현식은 변수 **x,y**를 여러번 수반하므로, backpropagation을 수행할 때 이 변수들에 대한 그라디언트 값을 축적하기 위해 `=` 대신 `+=`를 사용해야 하는 점에 주의해야 한다 (그렇게 하지 않으면 덮어쓰게 된다). 이는 Calculus에 나오는 *다변수 연쇄 법칙(multivariate chain rule)*을 따른다, Calculus에는 하나의 변수가 회로의 다른 부분들로 가지를 뻗어나가면, 반환하는 그라디언트는 더해질 것이라고 명시되어 있다.

<a name='patterns'></a>
### Patterns in backward flow

It is interesting to note that in many cases the backward-flowing gradient can be interpreted on an intuitive level. For example, the three most commonly used gates in neural networks (*add,mul,max*), all have very simple interpretations in terms of how they act during backpropagation. Consider this example circuit:

<div class="fig figleft fighighlight">
<svg width="460" height="290"><g transform="scale(1)"><defs><marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="50" y1="30" x2="90" y2="30" stroke="black" stroke-width="1"></line><text x="55" y="24" font-size="16" fill="green">3.00</text><text x="55" y="47" font-size="16" fill="red">-8.00</text><text x="45" y="24" font-size="16" text-anchor="end" fill="black">x</text><line x1="50" y1="100" x2="90" y2="100" stroke="black" stroke-width="1"></line><text x="55" y="94" font-size="16" fill="green">-4.00</text><text x="55" y="117" font-size="16" fill="red">6.00</text><text x="45" y="94" font-size="16" text-anchor="end" fill="black">y</text><line x1="50" y1="170" x2="90" y2="170" stroke="black" stroke-width="1"></line><text x="55" y="164" font-size="16" fill="green">2.00</text><text x="55" y="187" font-size="16" fill="red">2.00</text><text x="45" y="164" font-size="16" text-anchor="end" fill="black">z</text><line x1="50" y1="240" x2="90" y2="240" stroke="black" stroke-width="1"></line><text x="55" y="234" font-size="16" fill="green">-1.00</text><text x="55" y="257" font-size="16" fill="red">0.00</text><text x="45" y="234" font-size="16" text-anchor="end" fill="black">w</text><line x1="170" y1="65" x2="210" y2="65" stroke="black" stroke-width="1"></line><text x="175" y="59" font-size="16" fill="green">-12.00</text><text x="175" y="82" font-size="16" fill="red">2.00</text><circle cx="130" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="75" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="30" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="100" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="65" x2="170" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="170" y1="205" x2="210" y2="205" stroke="black" stroke-width="1"></line><text x="175" y="199" font-size="16" fill="green">2.00</text><text x="175" y="222" font-size="16" fill="red">2.00</text><circle cx="130" cy="205" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="210" font-size="20" fill="black" text-anchor="middle">max</text><line x1="90" y1="170" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="240" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="205" x2="170" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="290" y1="135" x2="330" y2="135" stroke="black" stroke-width="1"></line><text x="295" y="129" font-size="16" fill="green">-10.00</text><text x="295" y="152" font-size="16" fill="red">2.00</text><circle cx="250" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="250" y="140" font-size="20" fill="black" text-anchor="middle">+</text><line x1="210" y1="65" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="210" y1="205" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="270" y1="135" x2="290" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="410" y1="135" x2="450" y2="135" stroke="black" stroke-width="1"></line><text x="415" y="129" font-size="16" fill="green">-20.00</text><text x="415" y="152" font-size="16" fill="red">1.00</text><circle cx="370" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="370" y="140" font-size="20" fill="black" text-anchor="middle">*2</text><line x1="330" y1="135" x2="350" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="390" y1="135" x2="410" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></g></svg>
<div class="figcaption">
  An example circuit demonstrating the intuition behind the operations that backpropagation performs during the backward pass in order to compute the gradients on the inputs. Sum operation distributes gradients equally to all its inputs. Max operation routes the gradient to the higher input. Multiply gate takes the input activations, swaps them and multiplies by its gradient.
</div>
<div style="clear:both;"></div>
</div>

Looking at the diagram above as an example, we can see that:

The **add gate** always takes the gradient on its output and distributes it equally to all of its inputs, regardless of what their values were during the forward pass. This follows from the fact that the local gradient for the add operation is simply +1.0, so the gradients on all inputs will exactly equal the gradients on the output because it will be multiplied by x1.0 (and remain unchanged). In the example circuit above, note that the + gate routed the gradient of 2.00 to both of its inputs, equally and unchanged.

The **max gate** routes the gradient. Unlike the add gate which distributed the gradient unchanged to all its inputs, the max gate distributes the gradient (unchanged) to exactly one of its inputs (the input that had the highest value during the forward pass). This is because the local gradient for a max gate is 1.0 for the highest value, and 0.0 for all other values. In the example circuit above, the max operation routed the gradient of 2.00 to the **z** variable, which had a higher value than **w**, and the gradient on **w** remains zero.

The **multiply gate** is a little less easy to interpret. Its local gradients are the input values (except switched), and this is multiplied by the gradient on its output during the chain rule. In the example above, the gradient on **x** is -8.00, which is -4.00 x 2.00. 

*Unintuitive effects and their consequences*. Notice that if one of the inputs to the multiply gate is very small and the other is very big, then the multiply gate will do something slightly unintuitive: it will assign a relatively huge gradient to the small input and a tiny gradient to the large input. Note that in linear classifiers where the weights are dot producted $w^Tx_i$ (multiplied) with the inputs, this implies that the scale of the data has an effect on the magnitude of the gradient for the weights. For example, if you multiplied all input data examples $x_i$ by 1000 during preprocessing, then the gradient on the weights will be 1000 times larger, and you'd have to lower the learning rate by that factor to compensate. This is why preprocessing matters a lot, sometimes in subtle ways! And having intuitive understanding for how the gradients flow can help you debug some of these cases.

<a name='mat'></a>
### Gradients for vectorized operations

The above sections were concerned with single variables, but all concepts extend in a straight-forward manner to matrix and vector operations. However, one must pay closer attention to dimensions and transpose operations.

**Matrix-Matrix multiply gradient**. Possibly the most tricky operation is the matrix-matrix multiplication (which generalizes all matrix-vector and vector-vector) multiply operations:

~~~python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
~~~

*Tip: use dimension analysis!* Note that you do not need to remember the expressions for `dW` and `dX`  because they are easy to re-derive based on dimensions. For instance, we know that the gradient on the weights `dW` must be of the same size as `W` after it is computed, and that it must depend on matrix multiplication of `X` and `dD` (as is the case when both `X,W` are single numbers and not matrices). There is always exactly one way of achieving this so that the dimensions work out. For example, `X` is of size [10 x 3] and `dD` of size [5 x 3], so if we want `dW` and `W` has shape [5 x 10], then the only way of achieving this is with `dD.dot(X.T)`, as shown above.

**Work with small, explicit examples**. Some people may find it difficult at first to derive the gradient updates for some vectorized expressions. Our recommendation is to explicitly write out a minimal vectorized example, derive the gradient on paper and then generalize the pattern to its efficient, vectorized form.

<a name='summary'></a>
### Summary

- We developed intuition for what the gradients mean, how they flow backwards in the circuit, and how they communicate which part of the circuit should increase or decrease and with what force to make the final output higher.
- We discussed the importance of **staged computation** for practical implementations of backpropagation. You always want to break up your function into modules for which you can easily derive local gradients, and then chain them with chain rule. Crucially, you almost never want to write out these expressions on paper and differentiate them symbolically in full, because you never need an explicit mathematical equation for the gradient of the input variables. Hence, decompose your expressions into stages such that you can differentiate every stage independently (the stages will be matrix vector multiplies, or max operations, or sum operations, etc.) and then backprop through the variables one step at a time.

In the next section we will start to define Neural Networks, and backpropagation will allow us to efficiently compute the gradients on the connections of the neural network, with respect to a loss function. In other words, we're now ready to train Neural Nets, and the most conceptually difficult part of this class is behind us! ConvNets will then be a small step away.


### References

- [Automatic differentiation in machine learning: a survey](http://arxiv.org/abs/1502.05767)
