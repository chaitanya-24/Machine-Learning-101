# Math Operators
print(2+3*6)            #Output: '20'
print((2+3)*6)          #Output: '30'
print(2 ** 8)           #Output: '256'
print(23 // 7)          #Output: '3'
print(23 % 7)           #Output: '2'
print((5 - 1) * ((7 + 1) / (3 - 1)))            #Output: '16.0'


# Augmented Assignment Operators
greeting = 'Hello'
greeting += ' World!'
print(greeting)         #Output: 'Hello World!'

number = 1
number += 1
print(number)           #Output: '2'

mylist = ['item']
mylist *= 3
print(mylist)           #Output: ['item', 'item', 'item']


#Concatenation and Replication
print('Alice', 'Bob')   #Output: 'Alice Bob'
print('Alice' * 5)      #Output: 'AliceAliceAliceAliceAlice'


# The end keyword
phrase = ['printed', 'with', 'a', 'dash', 'in', 'between']
for word in phrase:
    print(word, end='-') 
print()           #Output: printed-with-a-dash-in-between-


#The sep keyword
print('cats', 'dogs', 'mice', sep=',')      #Output: 'cats,dogs,mice'


# len() Function
print(len('hello'))     #Output: '5'

print(len(['cat', 3, 'dog']))      #Output: '3'


a=[1,2,3]
if a:
    print("The list is not empty!")     #Output: 'The list is not empty!'


# str(), int(), and float() Functions
a = str(29)
print(a)       #Output: '29'
print(str(-3.14))       #Output: '-3.14'        

print(int('11'))        #Output: '11'
print(float('3.14'))    #Output: '3.14'

#The input() Function
print('What is your name?')
my_name = input()
print('Hi, {}'.format(my_name))
            #OR
my_name = input('What is your name? ')
print(f'Hi, {my_name}')

# Output: What is your name?
#         Martha
#         Hi, Martha





