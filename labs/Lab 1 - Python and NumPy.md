---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**PUT YOUR FULL NAME(S) HERE**


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
# YOUR SOLUTION HERE
#a=1000
import numpy as np
a = np.ones((6,4), dtype=np.int)*2
print(a)
```

## Exercise 2

```python
b = np.ones((6,4), dtype=np.int)
#np.fill_diagonal(b,3)
b[[0,1,2,3],[0,1,2,3]]=3
#note: inner arrays must be same size
print(b)
```

## Exercise 3

```python
c = a*b
print(c)

```

```python
# YOUR SOLUTION HERE
The multiplication operation works differently than the dot operation. The multiplication by default works
by multiplying each element in one matrix by the corresponding element in the other matrix. Since 
the matrices are the same size, this operation will work. The dot product requires that the number of columns 
in the first matrix match the number of rows in the second matrix.
```

## Exercise 4

```python
import numpy as np
transpA = np.dot(np.transpose(a), b)
print(transpA)

transpB = np.dot(a, np.transpose(b))
print(transpB)# YOUR SOLUTION HERE

print("The matrix that results from a dot product has the number of rows of the first matrix and the number of columns of the second matrix. Thus, it matters whether the transposed matrix is in the first or second position. ")
```

## Exercise 5

```python
print("my name is output")
```

## Exercise 6

```python
a = np.random.randint(2,6,(4,6))
b = np.random.randint(1, 10,(2,3))
print('a = \n', a)
print('\nb = \n', b)
print('\nsum of a =', np.sum(a))
print('\nsum of b columns =', np.sum(b, axis =0))
print('\nsum of a rows=', np.sum(a, axis = 1))
print('\nmean of b = ', np.mean(b))

```

## Exercise 7

```python
a = np.random.randint(0,6,(3,6))

def countOnes(arr):
    count =0
    print(arr)
    for x in range(len(arr)):
        for y in range(len(arr[x])):
            if arr[x][y] == 1:
                count+=1
    print('counting ones with loop= ', count)
   
countOnes(a)

b = np.where(a == 1, 1, 0)
print('counting ones with np.where=',np.sum(b))
   
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
import pandas as pd
a1 = [[2,2,2,2], [2,2,2,2], [2,2,2,2], [2,2,2,2], [2,2,2,2],[2,2,2,2]]
a = pd.DataFrame(a1, dtype =int)
a

a = pd.DataFrame(2, index=np.arange(6), columns=np.arange(4))
a
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python

b = pd.DataFrame(1, index=np.arange(6), columns=np.arange(4))
np.fill_diagonal(b.values,3)
b


```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
c = a * b
print(c)
```

## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
a = pd.DataFrame(np.random.randint(0, 3, size = (3, 4)))
print('arr =\n', a)
b = a.where(a==1)
count = np.sum(b.count())
print('number of ones =', count)
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv", delimiter=','
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df['name']
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv", delimiter=','
)
titanic_df.set_index('sex', inplace=True)
df = titanic_df.loc['female']
df

```

## Exercise 14
How do you reset the index?

```python
titanic_df.reset_index(inplace=True)
titanic_df
```

```python

```
