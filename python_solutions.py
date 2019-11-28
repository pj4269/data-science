import pandas as pd
import numpy as np
#import quandl
import datetime
#import MySQLdb
#from sqlalchemy import create_engine
import math

# https://courses.thinkful.com/dsbc-fundamentals-v1/checkpoint/8

# Numpy
#purchases.groupby('country').aggregate(np.mean) prob not necessary



# lesson 6: 

# puzzlebox

import random


list1 = [["Grae Drake", 98110], ["Bethany Kok"], ["Alex Nussbacher", 94101], ["Darrell Silver", 11201]]

dict1 = {t[0]: t[1:]   for t in list1}

print dict1
def user_contacts(list1):
  list3 = [None, None, None, None]
  list2 = []
  for i in list1:
    list2.append(i[0])

  for i in list1:
    # print i[1]
    if not i[1]:
       print 'var is False'
    #list3.append(i[1])
  #  try:
  #    list3.append(i[1])
  #  except: #i[1]==None:
  #    list3.append(None)

  dict1 = dict(zip(list2, list3))
  print dict1

#user_contacts(list1)


class Puzzlebox(object):
    """Puzzlebox for codewars kata."""

    def __init__(self):
        self.key = random.randint(1, 99)

    answer = "Ha, not quite that easy.\n"

    hint = "How do you normally unlock things?\n"
        
    hint_two = "The lock attribute is a method. Have you called it with anything yet?\n"

    def lock(self, *args):
        if len(args) != 1:
            return "This method expects one argument.\n"
        elif type(args[0]) != int:
            return "This method expects an integer.\n"
        elif args[0] != self.key:
            return "Did you know the key changes each time you print it?\n"
        else:
           
            return self.key, "You put the key in the lock! The answer is, of course, 42. Return that number from your answer() function to pass this kata.\n"

    def __repr__(self):
        return "The built-in dir() function is useful. Continue adding print statements till you know the answer.\n"

#puzzlebox = Puzzlebox()
#print(dir(puzzlebox))
'''
print puzzlebox.lock(puzzlebox.key)

def answer(puzzlebox):
  return  42

print answer(puzzlebox)
'''
# Vector
'''
class Vector(object):
 def __init__(self, a, b):
   self.a = a
   self.b = b
  
 def x(self):
   print self.a
 def y(self):
   print self.b

 def add(self, V):
   #self.a = self.a + V.a
   #self.b = self.b + V.b
   return Vector(self.a + V.a, self.b + V.b)


a = Vector(3, 4)
a.x()
a.y()

#print Vector(1, 2)
b = Vector(1, 2)

c = a.add(b)
c.x()
c.y()
'''

class Employee:
   empCount = 0

   def __init__(self, name, salary):
      self.name = name
      self.salary = salary
      Employee.empCount += 1
   
   def displayCount(self):
     print "Total Employee %d" % Employee.empCount

   def displayEmployee(self):
      print "Name : ", self.name,  ", Salary: ", self.salary
'''
emp1 = Employee("Zara", 2000)
emp2 = Employee("Manni", 5000)
emp1.displayEmployee()
emp2.displayEmployee()
'''
#print "Total Employee %d" % Employee.empCount

#3
#a.y

# lesson 5: dictionary
# 3.
def modes(data): 
  print list(data) 
  i = 0
  #while True:
    #print data[i]
    #i +=1
  list1 =  list(set(list(data)))
  list2 = []
  for i in list1:
    list2.append(list(data).count(i))
  dict1 =  dict(zip(list1,list2))  
  max_num = max(list2)
  list3 = []
  for k, s in dict1.items():
    if s==max_num:
      list3.append(k)
    else:
      pass
  if len(list3) == len(list1):
    return []
  else:
    return list3     

  #print dict1[max_num]

#print modes([1, 2, 2, 4])

#modes() that will accept one argument data
# 2. use contacts
'''
list1 = [["Grae Drake", 98110], ["Bethany Kok"], ["Alex Nussbacher", 94101], ["Darrell Silver", 11201]]
def user_contacts(list1):
  list2 = []
  list3 = []
  for i in list1:
    list2.append(i[0])
    try:
      list3.append(i[1])
    except: #i[1]==None:
      list3.append(None)    
  dict1  = dict(zip(list2, list3))
  print dict1

user_contacts(list1)
'''
# 1. order filler: 

#You've decided to write a function fillable() that takes three arguments: a dictionary stock representing all the merchandise you have in #stock, a string merch representing the thing your customer wants to buy, and an integer n representing the number of units of merch they #would like to buy. Your function should return True if you have the merchandise in stock to complete the sale, otherwise it should return #False.
stock =  {'a':10, 'b':20, 'c':0}
print stock
def fillable(stock, merch, n):
  if merch in stock:
    if n <= stock[merch] and n>=0: 
      return True
    else:
      return False
  else:
    return False  
print fillable(stock, 'd',1)


'''
pas = ''
while pas != 'unt':
   hi = input('try again?')
   if pas=='one':
      break
   else:
      continue
'''
# While loop 
#These while loops will look at a logical condition at each iteration and keep iterating (potentially forever) as long as that logical #condition is true
#while n % 2 == 0:
#    n = n // 2
#    print n

# looping through dict
#for a, b in dict1.items():
#  print a, b


def inverse_slice(list1, a, b): 
   print (list1, list1[a:b])
   for i in list1[a:b]:
      list1.remove(i)
   print list1
#inverse_slice([12, 14, 63, 72, 55, 24], 2, 4)
# Pop works with index
#fruits = ['apple', 'banana', 'cherry']
#fruits.pop(1)
#print(fruits)

# list of lists
'''
    [2, 5] --> 2 - 5 --> -3
    [3, 4] --> 3 - 4 --> -1
    [8, 7] --> 8 - 7 --> 1
'''
data = [[2, 5], [3, 4], [8, 7]]

def process_data(data):
   data1= [] 
   for i in data:
      data1.append(i[0]-i[1] )
   k = 1
   for z in data1:
      k *=z
   print k
#process_data(data)
      
# grade calculator
#print sum([92, 94, 99])/len([92, 94, 99])
# List: longest word
list1 = ['simple', 'is', 'better', 'than', 'complex']# ==> 7
def longest(word):
   list1 = ['simple', 'is', 'better', 'than', 'complex'] #==> 7
   list1.append(word)
   list2 = []
   for i in list1: 
      list2.append(len(i))
   dict1 = dict(zip(list2, list1))
   max1 = max(list2)
   return dict1[max1], max1
#print longest('hhhhhhhhhhhhhhhhh')
# P hackers
                     
def categorize_study(p, author_req):
   if author_req == 6: 
      bs = 1
   elif author_req == 5: 
      bs = 2
   elif author_req == 4: 
      bs = 4
   elif author_req == 3: 
      bs = 8
   # assuming
   elif author_req == 2: 
      bs = 16
   elif author_req == 1: 
      bs = 32
   elif author_req == 0: 
      bs = 64
   else: 
      pass
   
   product = bs * p
   #You've also decided that all studies meeting none of the author requirements that would have been categorized as "Fine" should instead 
   #be categorized as "Needs review".

   if product < 0.05:
      if author_req == 0:
         return 'Needs review' 
      else: 
         return "Fine"
   elif product >= 0.05 and product <= 0.15:
      return "Needs review"
   elif product > 0.15:
      return "Pants on fire"
   else: 
      return 'Something worn'

#print categorize_study(0.01, 3)# should return "Needs review" because the p-value times the bs-factor is 0.08.
#print categorize_study(0.04, 6)# should return "Fine" because the p-value times the bs-factor is only 0.04.
#print categorize_study(0.0001, 0)# should return "Needs review" even though the p-value times the bs-factor is only 0.0064.
#print categorize_study(0.012, 0)# should return "Pants on fire" because the p-value times the bs-factor is 0.768.
'''
    1 smooth red marble
    4 bumpy red marbles
    2 bumpy yellow marbles
    1 smooth yellow marble
    1 bumpy green marble
    1 smooth green marble
'''
def color_probability(col, bum): 
   if bum == 'bumpy':
      if col == 'red':
         print 'hi'
         return str(float(4)/float(7))[:4]
   elif col =='smooth':
      return ''
   else: 
      return ''

#print color_probability('red', 'bumpy')


# take_umbrella('sunny', 0.40) should return False.
def take_umbrella(w, num=0.1):
   if w == 'rainy' or num>=0.4:
      return True
   elif w=='sunny' and num>=0.5:
      return True
   else:
      return False



#print take_umb('sunny', 0.3)
# box_capacity(32, 64, 16) should return 13824.
def box_capacity(a, b, c):# should return 13824.
   return a*b*c* (12 **3)//(16**3)
#print box_capacity(32, 64, 16)# should return 13824.
#number problem: Blue and red marbles
def guess_blue(b, r, b_1, r_1): #0.6.
   return math.sqrt(float(float(b-b_1)/ float(b+r-b_1-r_1)))


def celsius_to_romer(C): 
  R = C * float(21)/float(40) + 7.5
  return R

#print celsius_to_romer(24)

















# 3. guessBlue(5, 5, 2, 3)

def guessBlue(start_b, start_r, pulled_b, pulled_r):
  prob = (float(start_b) - float( pulled_b) )/  (float(start_b)-float( pulled_b) + float(start_r) - float( pulled_r) )
  #prob = (start_b - pulled_b )/ ( start_b - pulled_b + start_r - pulled_r )
  return prob


print (guessBlue(5, 5, 2, 3) )

#guessBlue() should return the probability of drawing a blue marble, expressed as a float. For example, guessBlue(5, 5, 2, 3) should return 0.6.



# 2. Pixel Art Planning

def is_divisible(num1, num2):
  if num1 % num2 == 0: 
    return True
  else: 
    return False
  
#print (is_divisible(4050, 27))






# 1. Celsius to Romer
def c_t_r(celsius):

  Romer = celsius * 21/40 + 7.5

  return Romer

#print ( c_t_r(24) ) # 20.1



