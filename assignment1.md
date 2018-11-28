---
layout: page
mathjax: true
permalink: /assignment1/
---

이번 과제를 통해 간단한 k-Nearest Neighbor나 SVM/Softmax classifier에 기반한 image classification pipeline을 합치는 것을 연습하게 됩니다. 이번 과제의 목표는 다음과 같습니다.

- 간단한 **Image Classification pipeline** 이해 및 자료 주도적 접근 (data-driven approach, train/predict 단계에서) 이해
- 학습/검증/테스트로 데이터를 ***나누고***, **hyperparameter tuning**에 검증 데이터(validation data)를 사용하는 것을 이해
- numpy를 이용하여 **벡터화된** 효율적인 코드 작성 숙달
- k-Nearest Neighbor (**kNN**) classifier 구현 및 적용
- Multiclass Support Vector Machine (**SVM**) classifier 구현 및 적용
- **Softmax** classifier 구현 및 적용
- 위 classifier 간의 차이점과 장단점 이해
- 실제 픽셀 (예: 색상 히스토그램, 기울기 히스토그램 (HOG) 등) 대신에 **higher-level representations**을 사용하면 성능이 향상된다는 것을 간단히 이해

## 설정
이 과제는 로컬 머신에서 직접 실행하거나, [Terminal](https://www.terminal.com/)을 통해 가상 머신에서 수행하는 두 가지 방법으로 진행할 수 있습니다.

※ 이 과제에 필요한 모든 자료는 [여기](/assignments2016/assignment1)에서 받으실 수 있습니다.

## 로컬 머신에서 직접 실행

**코드 가져오기**

[Starter Code 다운로드](http://vision.stanford.edu/teaching/cs231n/assignment1.zip).

**[선택] virtual environment 설정**

Starter Code를 압축 해제한 뒤에 [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)을 만들 수도 있습니다. Visual environment를 사용하지 않으면 직접 본인 컴퓨터에 모든 dependency들을 설치해야 합니다.

visual environment 설정 방법은 아래와 같습니다.
~~~bash
cd assignment1
sudo pip install virtualenv      # 아마 설치되어 있을 것임
virtualenv .env                  # virtual environment 생성
source .env/bin/activate         # virtual environment 활성화
pip install -r requirements.txt  # dependency 설치
# Work on the assignment for a while ...deactivate                       # virtual environment에서 나가기
~~~

**데이터 다운로드**

starter code를 다운로드 하면, CIFAR-10 데이터셋도 다운로드 해야 합니다.
`assignment1` 디렉토리에서 다음을 실행합니다.

~~~bash
cd cs231n/datasets
./get_datasets.sh
~~~

**IPython 시작**
CIFAR-10 데이터를 받은 뒤, `assignment1` 디렉토리에서 IPython notebook server를 실행해야 합니다. IPython에 대한 자세한 정보는 [IPython tutorial](/ipython-tutorial)에서 확인하실 수 있습니다.

### 터미널에서 작업하기
이 과제를 위한 Terminal snapshot이 준비되어 있습니다. [여기](https://www.terminal.com/tiny/hUxP8UTMKa)에서 다운로드 가능합니다. Terminal을 이용하면 브라우저에서 과제를 수행할 수 있습니다. Terminal 튜토리얼은 [여기](/terminal-tutorial)서 확인하시기 바랍니다.

### 과제 제출
과제를 모두 마쳤다면 `collectSubmission.sh` 스크립트를 실행하십시오. 로컬 머신에서 직접 과제를 했든, Terminal을 이용했든 상관 없습니다. 이 파일을 이 과정을 위한 [coursework](https://coursework.stanford.edu/portal/site/W15-CS-231N-01/) 페이지를 통해 dropbox에 업로드하십시오.


### Q1: k-Nearest Neighbor classifier (30점)

IPython Notebook **knn.ipynb**을 이용해 kNN classifier를 구현합니다.

### Q2: Training a Support Vector Machine (30점)

IPython Notebook **svm.ipynb**을 이용해 SVM classifier를 구현합니다.

### Q3: Implement a Softmax classifier (30점)

IPython Notebook **softmax.ipynb**을 이용해 Softmax classifier를 구현합니다.

### Q4: Higher Level Representations: Image Features (10점)

IPython Notebook **features.ipynb**을 이용해 이 과제를 해결할 수 있습니다. raw pixel 값 대신에 higher-level representation을 이용하면 얻을 수 있는 향상을 볼 수 있을 것입니다.

### Q5: Bonus: 자신만의 feature 설계하기! (+10 점)

이 과제를 위한 Color histogram과 HOG feature가 제공됩니다. 이 보너스 점수를 얻으려면, numpy나 scipy (추가적인 dependency 사용 금지)만을 이용하여 자신만의 feature를 처음부터 구현하십시오. 구현하고 싶은 아이디어를 얻으려면 다른 feature가 무엇이 있는지 조사할 필요가 있습니다. 이 과제를 통해 만든 새로운 feature는 Q4의 feature보다 좋은 성능을 내야 보너스 점수를 바을 수 있습니다. 좋은 feature를 만들어 온다면 강의 시간에 소개하겠습니다.

### Q6: Cool Bonus: 무언가 더 해보기! (+10 점)

이 과제 주제와 관련된 다른 무언가를 구현하고, 조사하고, 분석한 뒤, 본인이 개발한 코드를 사용하십시오. 예를 들어, 질문을 받을 만한 다른 흥미로운 질문이 있을 것입니다. 이해하기 쉬운 시각화를 그려 볼 수 있지 않을까요? 아니면 loss function을 실험해 보는 것은 어떨까요? 끝내 주는 결과를 제출하면 보너스 점수를 받고, 수업 시간에 여러분의 결과를 소개해 줄 수도 있습니다.
