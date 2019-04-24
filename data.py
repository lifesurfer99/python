import operator

def add_new_entry():
    student = {}
    student['index'] = len(students) - 1
    student['id'] = int(input("id 입력: "))
    student['name'] = input("이름 입력: ")
    student['birth'] = input("입력: ")
    student['mid'] = int(input("중간고사 점수 입력: "))
    student['final'] = int(input("기말고사 점수 입력: "))

    students.append(student)

def delete_entry(key):
    stu_id = key
    index = 0
    for student in students:
        if student['id'] == stu_id:
            index = student['index']
            students.remove(student)
            break
    print(index)
    for student in students:
        if student['index'] > index:
            student['index'] -= 1
    return students

def find_item_entry(key):
    stu_id = key
    for student in students:
        if student['id'] == stu_id:
            student['total'] = student['mid'] + student['final']
            student['avg'] = student['total'] / 2
    print(student['total'], student['avg'])

def modify_entry(id, exam_type):
    stu_id = id

    flag = 1
    exam = exam_type
    if exam == "중간시험":
        flag = 0
    elif exam == "기말시험":
        flag = 1

    print(flag)

    for student in students:
        if student['id'] == stu_id and flag == 0:
            student['mid'] = int(input("중간시험 점수 입력: "))
            break
        elif student['id'] == stu_id and flag == 1:
            student['final'] = int(input("기말시험 점수 입력: "))
            break

    print(student['id'],'의 변경된 점수 확인', student)

def print_entry():
    for student in students:
            total = student['mid'] + student['final']
            student['avg'] = total / 2
            if student['avg'] >= 90:
                student['grade'] = 'A'
            elif student['avg'] < 90 and student['avg'] >= 80:
                student['grade'] = 'B'
            elif student['avg'] < 80 and student['avg'] >= 70:
                student['grade'] = 'C'
            elif student['avg'] < 70 and student['avg'] >= 60:
                student['grade'] = 'D'
            elif student['avg'] < 60 and student['avg'] >= 30:
                student['grade'] = 'E'
            elif student['avg'] < 30 and student['avg'] >= 0:
                student['grade'] = 'F'
    return students

def sort_entry(type):
    sort_type = type
    students = print_entry()

    sorted_students = sorted(students, key=lambda x: x['avg'], reverse=True)
    print()
    if sort_type == 'n':
        result_students = sorted(students, key = lambda x : x['name'])
        for student in result_students:
            print(student)
    elif sort_type == 'a':
        result_students = sorted(students, key=lambda x: x['avg'], reverse=True)
        for student in result_students:
            print(student)
    elif sort_type == 'g':
        result_students = sorted(students, key=lambda x: x['grade'])
        for student in result_students:
            print(student)

def quit_entry():
    exit()
def write_entry():
    result_students = print_entry()

    fp = open('./data.txt', 'w')

    for student in result_students:
        line = "\t".join([str(student['index']), str(student['id']), str(student['name']), str(student['birth']),
                         str(student['mid']), str(student['final']), str(student['avg']), str(student['grade']),
                         '\n'])
        fp.write(line)

    fp.close()

## main
f = open("data.txt", 'r', encoding='UTF-8')
lines = f.readlines()
f.close()
students = []

for line in lines:
    line = line.strip()
    items = line.split('\t')

    student = {}
    student['index'] = int(items[0])
    student['id'] = int(items[1])
    student['name'] = items[2]
    student['birth'] = items[3]
    student['mid'] = int(items[4])
    student['final'] = int(items[5])
    student['avg'] = 0.0
    student['grade'] = 0

    students.append(student)

while True:

    command = str(input("Choose one of the options below : "))

    if command == 'A' or command == 'a':
        add_new_entry()
        continue
    elif command == 'D' or command == 'd':
        student['id'] = int(input("학번을 입력하세요:"))
        id = student['id']
        delete_entry(id)
        continue
    elif command is 'F' or command == 'f':
        student['id'] = int(input("학번을 입력하세요:"))
        id = student['id']
        find_item_entry(id)
    elif command is 'M' or command == 'm':
        student['id'] = int(input("학번을 입력하세요:"))
        id = student['id']
        exam = input("중간시험 또는 기말시험 점수 중 어느 것을 수정하시겠습니까?: ")
        modify_entry(id, exam)
    elif command is 'P' or command == 'p':
        print(print_entry())
    elif command is 'S' or command == 's':
        type = input("이름순?(n), 평균점수순?(a), grade순?(g)")
        sort_entry(type)
    elif command is 'Q' or command == 'q':
        quit_entry()
    elif command is 'W' or command == 'w':
        write_entry()
