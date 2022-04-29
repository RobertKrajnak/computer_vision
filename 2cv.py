import numpy as np
import time
import numpy.matlib

# matica
A = np.array([[1,2],[3,4]])
print(A)

#riadkovy vektor

v_row = np.array([[1, 2, 3, 4]])
print(v_row)

#transponovanie
print(v_row.T)

#stplcovy vektor
v_column = np.array([[1,2,3,4]]).reshape((4,1))
print(v_column)

#zakladny datovy typ. 255 kvoli RGB farbe
np.array([], dtype=np.uint8)

#matica s nulami
zero = np.zeros((3, 4))
print(zero)

#jednotkova matica
one = np.ones((10,10))*255
print(one)

#diagonalna matica
diagonal = np.eye(3)
print(diagonal)

#random matica
random = np.random.rand(3, 4)*255
print(random)

#suma
vec_a = np.array([1, 2, 3, 4])
print(np.sum(vec_a))

#operacie s vektormi

vec_a = np.array([1, 2, 3])
vec_b = np.array([1, 2, 3])

#skalarny sucin
print(vec_a @ vec_b)

mat_1 = np.random.rand(3, 4)*255
mat_2 = np.random.rand(4, 4)*255

print(np.matmul(mat_1,mat_2))


vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])

print(np.concatenate(vec_a, vec_b))


########################################################



#Cyklus od 1 po 7 s krokom 2
for i in range(1,7,2):
  print(i)


for i in np.array([5,13,-1]):
    if i > 10:
        print("larger than ten")
    elif i < 0:
        print("negative value")
    else:
        print("something else")


# Implementácia s použitím cyklov
m=5
n=2
A = np.ones((m,n))
#print(A)
v = np.random.rand(1, n) * 2
#print(v)

for row in A:
    print(row - v)

print("---------")
#Implementácia s použitím maticových operácií
A = np.ones((m, n)) - numpy.matlib.repmat(v, m, 1);
print(A)

# Implementácia s použitím cyklov
B = np.zeros((m,n));
for i in range(m):
 for j in range(n):
    if A[i,j]>0:
        B[i,j] = A[i,j]
print(B)

print("-------------")
#Implementácia s použitím maticových operácií
ind = numpy.where(A > 0)
#print(ind)
B[ind] = A[ind]
print(B)

#ekvivalent to Matlab tic toc function
t = time.time()
#funcionality
elapsed = time.time() - t
print(elapsed)



# funkcia s jedným argumentom a jednou návratovou hodnotou
def myfunction(x):
    a = np.array([2,-1,0,1])
    y = a + x
    return y
#volanie funkcie
a = np.array([1, 2, 3, 4])
b = myfunction(2 * a)
#print(b)
# funkcia dvoch argumentov a s dvomi návratovými hodnotami
def myotherfunction(a,b):
     y = a + b
     z = a - b
     return y,z
#volanie funkcie
[c, d] = myotherfunction(a ,b)
#print(c)
#print(d)

