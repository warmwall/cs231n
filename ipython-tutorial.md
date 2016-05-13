---
layout: page
title: IPython Tutorial
permalink: /ipython-tutorial/
---
cs231s 수업에서는 프로그래밍 과제 진행을 위해 [IPython notebooks](http://ipython.org/)을 사용합니다. IPython notebook을 사용하면 여러분의 브라우저에서 Python 코드를 작성하고 실행할 수 있습니다. Python notebook를 사용하면 여러 조각의 코드를 아주 쉽게 수정하고 실행할 수 있습니다. 이런 장점 때문에 IPython notebook은 계산과학분야에서 널리 사용되고 있습니다.

IPython의 설치와 실행은 간단합니다. command line에서 다음 명령어를 입력하여 IPython을 설치합니다.

~~~
pip install "ipython[notebook]"
~~~

IPython의 설치가 완료되면 다음 명령어를 통해 IPython을 실행합니다.

~~~
ipython notebook
~~~

IPython이 실행되면, IPyhton을 사용하기 위해 웹 브라우저를 실행하여 http://localhost:8888 에 접속합니다. 모든 것이 잘 작동한다면 웹 브라우저에는 아래와 같은 화면이 나타납니다. 화면에는 현재 폴더에 사용가능한 Python notebook들이 나타납니다.

<div class='fig figcenter'>
  <img src='{{site.baseurl}}/assets/ipython-tutorial/file-browser.png'>
</div>

notebook 파일을 클릭하면 다음과 같은 화면이 나타납니다.

<div class='fig figcenter'>
  <img src='{{site.baseurl}}/assets/ipython-tutorial/notebook-1.png'>
</div>

IPython notebook은 여러 개의 **cell**들로 이루어져 있습니다. 각각의 cell들은 Python 코드를 포함하고 있습니다. `Shift-Enter`를 누르거나 셀을 클릭하여 셀을 실행할 수 있습니다. 셀의 코드를 실행하면 셀의 코드의 실행결과는 셀의 바로 아래에 나타납니다. 예를 들어 첫 번째 cell의 코드를 실행하면 아래와 같은 화면이 나타납니다.

<div class='fig figcenter'>
  <img src='{{site.baseurl}}/assets/ipython-tutorial/notebook-2.png'>
</div>

전역변수들은 다른 셀들에도 공유됩니다. 두 번째 셀을 실행하면 다음과 같은 결과가 나옵니다.

<div class='fig figcenter'>
  <img src='{{site.baseurl}}/assets/ipython-tutorial/notebook-3.png'>
</div>

일반적으로, IPython notebook의 코드를 실행할 때 맨 위에서 맨 아래 순서로 실행합니다.
몇몇 셀을 실행하는 데 실패하거나 셀들을 순서대로 실행하지 않으면 오류가 발생할 수 있습니다.

<div class='fig figcenter'>
  <img src='{{site.baseurl}}/assets/ipython-tutorial/notebook-error.png'>
</div>

과제를 진행하면서 notebook의 cell을 수정하거나 실행하여 IPython notebook이 변경되었다면 **저장하는 것을 잊지 마세요.**

<div class='fig figcenter'>
  <img src='{{site.baseurl}}/assets/ipython-tutorial/save-notebook.png'>
</div>

지금까지 IPyhton의 사용법에 대해서 알아보았습니다. 간략한 내용이지만 위 내용을 잘 숙지하면 무리 없이 과제를 진행할 수 있습니다.

---
<p style="text-align:right"><b>
번역: 김우정 <a href="https://github.com/gnujoow" style="color:black">(gnujoow)</a>
</b></p>
