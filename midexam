#2번 문제 - Pandas 추가 

import numpy as np
import pandas as pd
import re

def number_search():

    store_name = str(input("상호 입력: "))
    df = pd.read_csv('phonebook.txt', sep=' ', names=['store', 'phone'])
    print(df[df['store'].isin([store_name])])

def region_book():
    local = str(input("지역 국번 입력: "))
    df = pd.read_csv('phonebook.txt', sep=' ', names=['store', 'phone'])

    df['local_num'] = df.phone.map(lambda x: str(x).split('-')[0])
    print(df[df['local_num'].isin([local])].iloc[:, 0:2])
    pass

number_search()
region_book()

# 3번 문제 - numpy 추가 

import numpy as np

def matmul(A, B):

    A = A
    B = B
    C = np.matmul(A, B)
    return C

A = np.arange(4).reshape([2,2])
B = np.arange(4).reshape([2, 2])
print(matmul(A, B))

A = np.arange(6).reshape([2, 3])
B = np.arange(3).reshape([3, 1])

print(matmul(A, B))

# 4번 문제

def get_span(P):
    #  Input: N일 동안의 판매액을 저장한 List P
    n = int(input("리스트 요소 개수 입력: "))
    P = []
    for x in range(0, n):
        sale = int(input("요소 입력: "))
        P.append(sale)
    #  Input: N일 동안의 판매액을 저장한 List P
    #  Output: 각 날짜에 대한 span값을 저장한 배열 S
    S = []
    span = 1
    min = P[0]
    for val in P:
        if val > min:
            span = span + 1
            S.append(span)
        else:
            span = 1
            S.append(span)
    return S

x = []
print(get_span(x))
