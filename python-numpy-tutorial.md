---
layout: page
title: Python Numpy Tutorial
permalink: /python-numpy-tutorial/
---

<!--
Python:
  Simple data types
    integer, float, string
  Compound data types
    tuple, list, dictionary, set
  Flow control
    if, while, for, try, with
  Comprehensions, generators
  Functions
  Classes
  Standard library
    json, collections, itertools

Numpy
-->

이 튜토리얼은 [Justin Johnson](http://cs.stanford.edu/people/jcjohns/)에 의해 작성되었습니다.

cs231n 수업의 모든 과제에서는 프로그래밍 언어로 파이썬을 사용할 것입니다.
파이썬은 그 자체만으로도 훌륭한 범용 프로그래밍 언어이지만, 몇몇 라이브러리(numpy, scipy, matplotlib)의 도움으로 
계산과학 분야에서 강력한 개발 환경을 갖추게 됩니다.  

많은 분들이 파이썬과 numpy를 경험 해보셨을거라고 생각합니다. 경험 하지 못했을지라도 이 문서를 통해  
'프로그래밍 언어로서의 파이썬'과 '파이썬을 계산과학에 활용하는법'을 빠르게 훑을 수 있습니다.

만약 Matlab을 사용해보셨다면, [Matlab사용자를 위한 numpy](http://wiki.scipy.org/NumPy_for_Matlab_Users) 페이지를 추천해 드립니다.

또한 [CS 228](http://cs.stanford.edu/~ermon/cs228/index.html)수업을 위해 [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) 와 [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335)가 만든 [이 튜토리얼의 IPython notebook 버전](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb)도 참조 할 수 있습니다.

목차:

- [파이썬](#python)
  - [기본 자료형](#python-basic)
  - [컨테이너](#python-containers)
      - [리스트](#python-lists)
      - [딕셔너리](#python-dicts)
      - [집합](#python-sets)
      - [튜플](#python-tuples)
  - [함수](#python-functions)
  - [클래스](#python-classes)
- [Numpy](#numpy)
  - [배열](#numpy-arrays)
  - [배열 색인](#numpy-array-indexing)
  - [데이터타입](#numpy-datatypes)
  - [배열 연산](#numpy-math)
  - [브로드캐스팅](#numpy-broadcasting)
- [SciPy](#scipy)
  - [이미지 작업](#scipy-image)
  - [MATLAB 파일](#scipy-matlab)
  - [두 점 사이의 거리](#scipy-dist)
- [Matplotlib](#matplotlib)
  - [Plotting](#matplotlib-plotting)
  - [Subplots](#matplotlib-subplots)
  - [이미지](#matplotlib-images)

<a name='python'></a>
## Python
파이썬은 고차원이고, 다중패러다임을 지원하는 동적 프로그래밍 언어이다. 
짧지만 가독성 높은 코드 몇 줄로 수준 높은 아이디어들을 표현할수있기에 파이썬 코드는 거의 수도코드처럼 보인다고도 한다. 
아래는 quicksort알고리즘의 파이썬 구현 예시이다:

~~~python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) / 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
    
print quicksort([3,6,8,10,1,2,1])
# 출력 "[1, 1, 2, 3, 6, 8, 10]"
~~~

### 파이썬 버전
현재 파이썬에는 두가지 버전이 있습니다. 파이썬 2.7 그리고 파이썬 3.4입니다. 
혼란스럽게도, 파이썬3은 기존 파이썬2와 호환되지 않게 변경된 부분이 있습니다.
그러므로 파이썬 2.7로 쓰여진 코드는 3.4환경에서 동작하지 않고 그 반대도 마찬가지입니다.
이 수업에선 파이썬 2.7을 사용합니다.

커맨드라인에 아래의 명령어를 입력해서 현재 설치된 파이썬 버전을 확인 할 수 있습니다.
`python --version`.

<a name='python-basic'></a>
### 기본 자료형

다른 프로그래밍 언어들처럼, 파이썬에는 정수, 실수, 불린, 문자열같은 기본 자료형이 있습니다.
파이썬 기본 자료형 역시 다른 프로그래밍 언어와 유사합니다.

**숫자:** 다른 언어와 마찬가지로 파이썬의 정수형(Integers)과 실수형(floats) 데이터 타입 역시 동일한 역할을 합니다 :

~~~python
x = 3
print type(x) # 출력 "<type 'int'>"
print x       # 출력 "3"
print x + 1   # 덧셈; 출력 "4"
print x - 1   # 뺄셈; 출력 "2"
print x * 2   # 곱셈; 출력 "6"
print x ** 2  # 제곱; 출력 "9"
x += 1
print x  # 출력 "4"
x *= 2
print x  # 출력 "8"
y = 2.5
print type(y) # 출력 "<type 'float'>"
print y, y + 1, y * 2, y ** 2 # 출력 "2.5 3.5 5.0 6.25"
~~~
다른 언어들과는 달리, 파이썬에는 증감 단항연상자(`x++`, `x--`)가 없습니다.

파이썬 역시 long 정수형과 복소수 데이터 타입이 구현되어 있습니다. 
자세한 사항은 [문서](https://docs.python.org/2/library/stdtypes.html#numeric-types-int-float-long-complex)에서 찾아볼 수 있습니다.

**불린(Booleans):** 파이썬에는 논리 자료형의 모든 연산자들이 구현되어 있습니다. 
그렇지만 기호(`&&`, `||`, 등.) 대신 영어 단어로 구현되어 있습니다 :

~~~python
t = True
f = False
print type(t) # 출력 "<type 'bool'>"
print t and f # 논리 AND; 출력 "False"
print t or f  # 논리 OR; 출력 "True"
print not t   # 논리 NOT; 출력 "False"
print t != f  # 논리 XOR; 출력 "True" 
~~~

**문자열:** 파이썬은 문자열과 연관된 다양한 기능을 지원합니다:

~~~python
hello = 'hello'   # String 문자열을 표현할땐 따옴표나
world = "world"   # 쌍따옴표가 사용됩니다; 어떤걸 써도 상관없습니다.
print hello       # 출력 "hello"
print len(hello)  # 문자열 길이; 출력 "5"
hw = hello + ' ' + world  # 문자열 연결
print hw  # 출력 "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf 방식의 문자열 서식 지정
print hw12  # 출력 "hello world 12"
~~~

문자열 객체에는 유용한 메소드들이 많습니다; 예를 들어:

~~~python
s = "hello"
print s.capitalize()  # 문자열을 대문자로 시작하게함; 출력 "Hello"
print s.upper()       # 모든 문자를 대문자로 바꿈; 출력 "HELLO"
print s.rjust(7)      # 문자열 오른쪽 정렬, 빈공간은 여백으로 채움; 출력 "  hello"
print s.center(7)     # 문자열 가운데 정렬, 빈공간은 여백으로 채움; 출력 " hello "
print s.replace('l', '(ell)')  # 첫번째 인자로 온 문자열을 두번째 인자 문자열로 바꿈;
                               # 출력 "he(ell)(ell)o"
print '  world '.strip()  # 문자열 앞뒤 공백 제거; 출력 "world"
~~~
모든 문자열 메소드는 [문서](https://docs.python.org/2/library/stdtypes.html#string-methods)에서 찾아볼 수 있습니다. 

<a name='python-containers'></a>
### 컨테이너

파이썬은 다음과 같은 컨테이너 타입이 구현되어 있습니다: 리스트, 딕셔너리, 집합, 튜플

<a name='python-lists'></a>
#### 리스트

리스트는 파이썬에서 배열같은 존재입니다. 그렇지만 배열과 달리 크기 변경이 가능하고
서로 다른 자료형일지라도 하나의 리스트에 저장 될 수 있습니다:

~~~python
xs = [3, 1, 2]   # 리스트 생성
print xs, xs[2]  # 출력 "[3, 1, 2] 2"
print xs[-1]     # 인덱스가 음수일 경우 리스트의 끝에서부터 세어진다; 출력 "2"
xs[2] = 'foo'    # 리스트는 자료형이 다른 요소들을 저장 할 수 있다
print xs         # 출력 "[3, 1, 'foo']"
xs.append('bar') # 리스트의 끝에 새 요소 추가
print xs         # 출력 "[3, 1, 'foo', 'bar']"
x = xs.pop()     # 리스트의 마지막 요소 삭제하고 반환
print x, xs      # 출력 "bar [3, 1, 'foo']"
~~~
마찬가지로, 리스트에 대해 자세하 사항은 [문서](https://docs.python.org/2/tutorial/datastructures.html#more-on-lists)에서 찾아볼 수 있습니다.

**슬라이싱:**
리스트의 요소로 한번에 접근하는것 이외에도, 파이썬은 리스트의 일부분에만 접근하는 간결한 문법을 제공한다;
이를 *슬라이싱*이라고 한다:

~~~python
nums = range(5)    # range는 파이썬에 구현되어 있는 함수이며 정수들로 구성된 리스트를 만든다
print nums         # 출력 "[0, 1, 2, 3, 4]"
print nums[2:4]    # 인덱스 2에서 4(제외)까지 슬라이싱; 출력 "[2, 3]"
print nums[2:]     # 인덱스 2에서 끝까지 슬라이싱; 출력 "[2, 3, 4]"
print nums[:2]     # 처음부터 인덱스 2(제외)까지 슬라이싱; 출력 "[0, 1]"
print nums[:]      # 전체 리스트 슬라이싱; 출력 ["0, 1, 2, 3, 4]"
print nums[:-1]    # 슬라이싱 인덱스는 음수도 가능; 출력 ["0, 1, 2, 3]"
nums[2:4] = [8, 9] # 슬라이스된 리스트에 새로운 리스트 할당
print nums         # 출력 "[0, 1, 8, 9, 4]"
~~~
numpy 배열 부분에서 다시 슬라이싱을 보게될것입니다.

**Loops:** You can loop over the elements of a list like this:

~~~python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print animal
# 출력 "cat", "dog", "monkey", each on its own line.
~~~

If you want access to the index of each element within the body of a loop,
use the built-in `enumerate` function:

~~~python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)
# 출력 "#1: cat", "#2: dog", "#3: monkey", each on its own line
~~~

**List comprehensions:**
When programming, frequently we want to transform one type of data into another.
As a simple example, consider the following code that computes square numbers:

~~~python
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print squares   # 출력 [0, 1, 4, 9, 16]
~~~

You can make this code simpler using a **list comprehension**:

~~~python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print squares   # 출력 [0, 1, 4, 9, 16]
~~~

List comprehensions can also contain conditions:

~~~python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares  # 출력 "[0, 4, 16]"
~~~

<a name='python-dicts'></a>
#### Dictionaries
A dictionary stores (key, value) pairs, similar to a `Map` in Java or
an object in Javascript. You can use it like this:

~~~python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print d['cat']       # Get an entry from a dictionary; 출력 "cute"
print 'cat' in d     # Check if a dictionary has a given key; 출력 "True"
d['fish'] = 'wet'    # Set an entry in a dictionary
print d['fish']      # 출력 "wet"
# print d['monkey']  # KeyError: 'monkey' not a key of d
print d.get('monkey', 'N/A')  # Get an element with a default; 출력 "N/A"
print d.get('fish', 'N/A')    # Get an element with a default; 출력 "wet"
del d['fish']        # Remove an element from a dictionary
print d.get('fish', 'N/A') # "fish" is no longer a key; 출력 "N/A"
~~~
You can find all you need to know about dictionaries
[in the documentation](https://docs.python.org/2/library/stdtypes.html#dict).

**Loops:** It is easy to iterate over the keys in a dictionary:

~~~python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print 'A %s has %d legs' % (animal, legs)
# 출력 "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"
~~~

If you want access to keys and their corresponding values, use the `iteritems` method:

~~~python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.iteritems():
    print 'A %s has %d legs' % (animal, legs)
# 출력 "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"
~~~

**Dictionary comprehensions:**
These are similar to list comprehensions, but allow you to easily construct
dictionaries. For example:

~~~python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print even_num_to_square  # 출력 "{0: 0, 2: 4, 4: 16}"
~~~

<a name='python-sets'></a>
#### Sets
A set is an unordered collection of distinct elements. As a simple example, consider
the following:

~~~python
animals = {'cat', 'dog'}
print 'cat' in animals   # Check if an element is in a set; 출력 "True"
print 'fish' in animals  # 출력 "False"
animals.add('fish')      # Add an element to a set
print 'fish' in animals  # 출력 "True"
print len(animals)       # Number of elements in a set; 출력 "3"
animals.add('cat')       # Adding an element that is already in the set does nothing
print len(animals)       # 출력 "3"
animals.remove('cat')    # Remove an element from a set
print len(animals)       # 출력 "2"
~~~

As usual, everything you want to know about sets can be found
[in the documentation](https://docs.python.org/2/library/sets.html#set-objects).


**Loops:**
Iterating over a set has the same syntax as iterating over a list;
however since sets are unordered, you cannot make assumptions about the order
in which you visit the elements of the set:

~~~python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)
# 출력 "#1: fish", "#2: dog", "#3: cat"
~~~

**Set comprehensions:**
Like lists and dictionaries, we can easily construct sets using set comprehensions:

~~~python
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print nums  # 출력 "set([0, 1, 2, 3, 4, 5])"
~~~

<a name='python-tuples'></a>
#### Tuples
A tuple is an (immutable) ordered list of values.
A tuple is in many ways similar to a list; one of the most important differences is that
tuples can be used as keys in dictionaries and as elements of sets, while lists cannot.
Here is a trivial example:

~~~python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print type(t)    # 출력 "<type 'tuple'>"
print d[t]       # 출력 "5"
print d[(1, 2)]  # 출력 "1"
~~~
[The documentation](https://docs.python.org/2/tutorial/datastructures.html#tuples-and-sequences) has more information about tuples.

<a name='python-functions'></a>
### Functions
Python functions are defined using the `def` keyword. For example:

~~~python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print sign(x)
# 출력 "negative", "zero", "positive"
~~~

We will often define functions to take optional keyword arguments, like this:

~~~python
def hello(name, loud=False):
    if loud:
        print 'HELLO, %s!' % name.upper()
    else:
        print 'Hello, %s' % name

hello('Bob') # 출력 "Hello, Bob"
hello('Fred', loud=True)  # 출력 "HELLO, FRED!"
~~~
There is a lot more information about Python functions
[in the documentation](https://docs.python.org/2/tutorial/controlflow.html#defining-functions).

<a name='python-classes'></a>
### Classes

The syntax for defining classes in Python is straightforward:

~~~python
class Greeter(object):
    
    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable
        
    # Instance method
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name
        
g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; 출력 "Hello, Fred"
g.greet(loud=True)   # Call an instance method; 출력 "HELLO, FRED!"
~~~
You can read a lot more about Python classes
[in the documentation](https://docs.python.org/2/tutorial/classes.html).

<a name='numpy'></a>
## Numpy

[Numpy](http://www.numpy.org/) is the core library for scientific computing in Python.
It provides a high-performance multidimensional array object, and tools for working with these
arrays. If you are already familiar with MATLAB, you might find
[this tutorial useful](http://wiki.scipy.org/NumPy_for_Matlab_Users) to get started with Numpy.

<a name='numpy-arrays'></a>
### Arrays
A numpy array is a grid of values, all of the same type, and is indexed by a tuple of
nonnegative integers. The number of dimensions is the *rank* of the array; the *shape*
of an array is a tuple of integers giving the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists,
and access elements using square brackets:

~~~python
import numpy as np

a = np.array([1, 2, 3])  # Create a rank 1 array
print type(a)            # 출력 "<type 'numpy.ndarray'>"
print a.shape            # 출력 "(3,)"
print a[0], a[1], a[2]   # 출력 "1 2 3"
a[0] = 5                 # Change an element of the array
print a                  # 출력 "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print b.shape                     # 출력 "(2, 3)"
print b[0, 0], b[0, 1], b[1, 0]   # 출력 "1 2 4"
~~~

Numpy also provides many functions to create arrays:

~~~python
import numpy as np

a = np.zeros((2,2))  # Create an array of all zeros
print a              # 출력 "[[ 0.  0.]
                     #          [ 0.  0.]]"
    
b = np.ones((1,2))   # Create an array of all ones
print b              # 출력 "[[ 1.  1.]]"

c = np.full((2,2), 7) # Create a constant array
print c               # 출력 "[[ 7.  7.]
                      #          [ 7.  7.]]"

d = np.eye(2)        # Create a 2x2 identity matrix
print d              # 출력 "[[ 1.  0.]
                     #          [ 0.  1.]]"
    
e = np.random.random((2,2)) # Create an array filled with random values
print e                     # Might print "[[ 0.91940167  0.08143941]
                            #               [ 0.68744134  0.87236687]]"
~~~
You can read about other methods of array creation
[in the documentation](http://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation).

<a name='numpy-array-indexing'></a>
### Array indexing
Numpy offers several ways to index into arrays.

**Slicing:**
Similar to Python lists, numpy arrays can be sliced.
Since arrays may be multidimensional, you must specify a slice for each dimension
of the array:

~~~python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print a[0, 1]   # 출력 "2"
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print a[0, 1]   # 출력 "77"
~~~

You can also mix integer indexing with slice indexing.
However, doing so will yield an array of lower rank than the original array.
Note that this is quite different from the way that MATLAB handles array
slicing:

~~~python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print row_r1, row_r1.shape  # 출력 "[5 6 7 8] (4,)"
print row_r2, row_r2.shape  # 출력 "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print col_r1, col_r1.shape  # 출력 "[ 2  6 10] (3,)"
print col_r2, col_r2.shape  # 출력 "[[ 2]
                            #          [ 6]
                            #          [10]] (3, 1)"
~~~

**Integer array indexing:**
When you index into numpy arrays using slicing, the resulting array view
will always be a subarray of the original array. In contrast, integer array
indexing allows you to construct arbitrary arrays using the data from another
array. Here is an example:

~~~python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and 
print a[[0, 1, 2], [0, 1, 0]]  # 출력 "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print np.array([a[0, 0], a[1, 1], a[2, 0]])  # 출력 "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print a[[0, 0], [1, 1]]  # 출력 "[2 2]"

# Equivalent to the previous integer array indexing example
print np.array([a[0, 1], a[0, 1]])  # 출력 "[2 2]"
~~~

One useful trick with integer array indexing is selecting or mutating one
element from each row of a matrix:

~~~python
import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print a  # 출력 "array([[ 1,  2,  3],
         #                [ 4,  5,  6],
         #                [ 7,  8,  9],
         #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print a[np.arange(4), b]  # 출력 "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print a  # 출력 "array([[11,  2,  3],
         #                [ 4,  5, 16],
         #                [17,  8,  9],
         #                [10, 21, 12]])
~~~

**Boolean array indexing:**
Boolean array indexing lets you pick out arbitrary elements of an array.
Frequently this type of indexing is used to select the elements of an array
that satisfy some condition. Here is an example:

~~~python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.
            
print bool_idx      # 출력 "[[False False]
                    #          [ True  True]
                    #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print a[bool_idx]  # 출력 "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print a[a > 2]     # 출력 "[3 4 5 6]"
~~~

For brevity we have left out a lot of details about numpy array indexing;
if you want to know more you should
[read the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

<a name='numpy-datatypes'></a>
### Datatypes
Every numpy array is a grid of elements of the same type.
Numpy provides a large set of numeric datatypes that you can use to construct arrays.
Numpy tries to guess a datatype when you create an array, but functions that construct
arrays usually also include an optional argument to explicitly specify the datatype.
Here is an example:

~~~python
import numpy as np

x = np.array([1, 2])  # Let numpy choose the datatype
print x.dtype         # 출력 "int64"

x = np.array([1.0, 2.0])  # Let numpy choose the datatype
print x.dtype             # 출력 "float64"

x = np.array([1, 2], dtype=np.int64)  # Force a particular datatype
print x.dtype                         # 출력 "int64"
~~~
You can read all about numpy datatypes
[in the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).

<a name='numpy-math'></a>
### Array math
Basic mathematical functions operate elementwise on arrays, and are available
both as operator overloads and as functions in the numpy module:

~~~python
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print x + y
print np.add(x, y)

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print x - y
print np.subtract(x, y)

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print x * y
print np.multiply(x, y)

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print x / y
print np.divide(x, y)

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print np.sqrt(x)
~~~

Note that unlike MATLAB, `*` is elementwise multiplication, not matrix
multiplication. We instead use the `dot` function to compute inner
products of vectors, to multiply a vector by a matrix, and to
multiply matrices. `dot` is available both as a function in the numpy
module and as an instance method of array objects:

~~~python
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print v.dot(w)
print np.dot(v, w)

# Matrix / vector product; both produce the rank 1 array [29 67]
print x.dot(v)
print np.dot(x, v)

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print x.dot(y)
print np.dot(x, y)
~~~

Numpy provides many useful functions for performing computations on
arrays; one of the most useful is `sum`:

~~~python
import numpy as np

x = np.array([[1,2],[3,4]])

print np.sum(x)  # Compute sum of all elements; 출력 "10"
print np.sum(x, axis=0)  # Compute sum of each column; 출력 "[4 6]"
print np.sum(x, axis=1)  # Compute sum of each row; 출력 "[3 7]"
~~~
You can find the full list of mathematical functions provided by numpy
[in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html).

Apart from computing mathematical functions using arrays, we frequently
need to reshape or otherwise manipulate data in arrays. The simplest example
of this type of operation is transposing a matrix; to transpose a matrix, 
simply use the `T` attribute of an array object:

~~~python
import numpy as np

x = np.array([[1,2], [3,4]])
print x    # 출력 "[[1 2]
           #          [3 4]]"
print x.T  # 출력 "[[1 3]
           #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print v    # 출력 "[1 2 3]"
print v.T  # 출력 "[1 2 3]"
~~~
Numpy provides many more functions for manipulating arrays; you can see the full list
[in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html).


<a name='numpy-broadcasting'></a>
### Broadcasting
Broadcasting is a powerful mechanism that allows numpy to work with arrays of different
shapes when performing arithmetic operations. Frequently we have a smaller array and a
larger array, and we want to use the smaller array multiple times to perform some operation
on the larger array.

For example, suppose that we want to add a constant vector to each
row of a matrix. We could do it like this:

~~~python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print y
~~~

This works; however when the matrix `x` is very large, computing an explicit loop
in Python could be slow. Note that adding the vector `v` to each row of the matrix
`x` is equivalent to forming a matrix `vv` by stacking multiple copies of `v` vertically,
then performing elementwise summation of `x` and `vv`. We could implement this
approach like this:

~~~python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print vv                 # 출력 "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print y  # 출력 "[[ 2  2  4
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]"
~~~

Numpy broadcasting allows us to perform this computation without actually
creating multiple copies of `v`. Consider this version, using broadcasting:

~~~python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print y  # 출력 "[[ 2  2  4]
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]"
~~~

The line `y = x + v` works even though `x` has shape `(4, 3)` and `v` has shape
`(3,)` due to broadcasting; this line works as if `v` actually had shape `(4, 3)`,
where each row was a copy of `v`, and the sum was performed elementwise.

Broadcasting two arrays together follows these rules:

1. If the arrays do not have the same rank, prepend the shape of the lower rank array
   with 1s until both shapes have the same length.
2. The two arrays are said to be *compatible* in a dimension if they have the same
   size in the dimension, or if one of the arrays has size 1 in that dimension.
3. The arrays can be broadcast together if they are compatible in all dimensions.
4. After broadcasting, each array behaves as if it had shape equal to the elementwise
   maximum of shapes of the two input arrays.
5. In any dimension where one array had size 1 and the other array had size greater than 1,
   the first array behaves as if it were copied along that dimension

If this explanation does not make sense, try reading the explanation
[from the documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
or [this explanation](http://wiki.scipy.org/EricsBroadcastingDoc).

Functions that support broadcasting are known as *universal functions*. You can find
the list of all universal functions
[in the documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).

Here are some applications of broadcasting:

~~~python
import numpy as np

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print np.reshape(v, (3, 1)) * w

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print x + v

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print (x.T + w).T
# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print x + np.reshape(w, (2, 1))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print x * 2
~~~

Broadcasting typically makes your code more concise and faster, so you
should strive to use it where possible.

### Numpy Documentation
This brief overview has touched on many of the important things that you need to
know about numpy, but is far from complete. Check out the
[numpy reference](http://docs.scipy.org/doc/numpy/reference/)
to find out much more about numpy.

<a name='scipy'></a>
## SciPy
Numpy provides a high-performance multidimensional array and basic tools to
compute with and manipulate these arrays.
[SciPy](http://docs.scipy.org/doc/scipy/reference/)
builds on this, and provides
a large number of functions that operate on numpy arrays and are useful for
different types of scientific and engineering applications.

The best way to get familiar with SciPy is to
[browse the documentation](http://docs.scipy.org/doc/scipy/reference/index.html).
We will highlight some parts of SciPy that you might find useful for this class.

<a name='scipy-image'></a>
### Image operations
SciPy provides some basic functions to work with images.
For example, it has functions to read images from disk into numpy arrays,
to write numpy arrays to disk as images, and to resize images.
Here is a simple example that showcases these functions:

~~~python
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print img.dtype, img.shape  # 출력 "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
~~~

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/cat.jpg'>
  <img src='{{site.baseurl}}/assets/cat_tinted.jpg'>
  <div class='figcaption'>
    Left: The original image.
    Right: The tinted and resized image.
  </div>
</div>

<a name='scipy-matlab'></a>
### MATLAB files
The functions `scipy.io.loadmat` and `scipy.io.savemat` allow you to read and
write MATLAB files. You can read about them
[in the documentation](http://docs.scipy.org/doc/scipy/reference/io.html).

<a name='scipy-dist'></a>
### Distance between points
SciPy defines some useful functions for computing distances between sets of points.

The function `scipy.spatial.distance.pdist` computes the distance between all pairs
of points in a given set:

~~~python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print x

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print d
~~~
You can read all the details about this function
[in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).

A similar function (`scipy.spatial.distance.cdist`) computes the distance between all pairs
across two sets of points; you can read about it
[in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).

<a name='matplotlib'></a>
## Matplotlib
[Matplotlib](http://matplotlib.org/) is a plotting library. 
In this section give a brief introduction to the `matplotlib.pyplot` module,
which provides a plotting system similar to that of MATLAB.

<a name='matplotlib-plot'></a>
### Plotting
The most important function in matplotlib is `plot`,
which allows you to plot 2D data. Here is a simple example:

~~~python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.
~~~

Running this code produces the following plot:

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/sine.png'>
</div>

With just a little bit of extra work we can easily plot multiple lines
at once, and add a title, legend, and axis labels:

~~~python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
~~~
<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/sine_cosine.png'>
</div>

You can read much more about the `plot` function
[in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot).

<a name='matplotlib-subplots'></a>
### Subplots
You can plot different things in the same figure using the `subplot` function.
Here is an example:

~~~python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
~~~

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/sine_cosine_subplot.png'>
</div>

You can read much more about the `subplot` function
[in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot).

<a name='matplotlib-images'></a>
### Images
You can use the `imshow` function to show images. Here is an example:

~~~python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
~~~

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/cat_tinted_imshow.png'>
</div>
