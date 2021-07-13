class Person:
    def __init__(self, age, height, bloodtype):
        print('constructor called!')
        self.age = age
        self.height = height
        self.bloodtype = bloodtype


    def print_person_info(self):
        print(f'age: {self.age}')
        print(f'height: {self.height}')
        print(f'b-type: {self.bloodtype}')

class Student(Person):
    def __init__(self, age, height, bloodtype, math, eng, python):
        super().__init__(age, height, bloodtype)
        self.math = math
        self.eng = eng
        self.python = python
    def print_student_info(self):
        self.print_person_info()
        print(f'math:{self.math}')
        print(f'eng:{self.eng}')

me = Student(33, 177, 'o', 100,100,100)
me.print_student_info()