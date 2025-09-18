import numpy as np
import pandas as pd
from fontTools.misc.cython import returns
from matplotlib import pyplot as plt
# Graphics in SVG format are more sharp and legible

pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.options.display.expand_frame_repr = False
pd.options.mode.copy_on_write = True

data = pd.read_csv('titanic_train.csv',
                  index_col='PassengerId')

def male_female(sex):
    if sex == "male":
        return 1
    return 0
def printLineSpace():
    print(8 * '-')
def q1():
    print("q1")
    sex_categories = [male_female(sex) for sex in data.Sex]
    data['sex_categories'] = sex_categories
    print(f"male {len(data[(data['sex_categories'] == 1)])}")
    print(f"female {len(data[(data['sex_categories'] == 0)])}")
def q2():
    print("q2")
    print(f"male in 2 class: {len(
        (data[(data['Pclass'] == 2) &
              (data['sex_categories'] == 1)]))}")
    print(f"female in 2 class: {len(
        (data[(data['Pclass'] == 2) &
              (data['sex_categories'] == 0)]))}")
    print(f"all people in 2 class: {len(data[(data['Pclass'] == 2)])}")
def q3():
    print("q3")
    print(round(data.describe()['Fare']['50%'], 2))
    print(round(data.describe()['Fare']['std'], 2))
def q4():
    print("q4")
    survived_ = data[data['Survived'] == 1]
    dead = data[data['Survived'] == 0]
    print(survived_.describe()['Age']['mean'] > dead.describe()['Age']['mean'])
def q5():
    print("q5")
    younger_30 = data[data['Age'] < 30]
    older_60 = data[data['Age'] > 60]
    print(len(younger_30[younger_30.Survived == 1]) > len(older_60[older_60.Survived == 1]))
    print(round(len(younger_30[younger_30.Survived == 1]) / len(younger_30) * 100, 1))
    print(round(len(older_60[older_60.Survived == 1]) / len(older_60) * 100, 1))
def q6():
    print("q6")
    man = data[data['sex_categories'] == 1]
    woman = data[data['sex_categories'] == 0]
    print(round(len(man[man.Survived == 1]) / len(man) * 100, 1))
    print(round(len(woman[woman.Survived == 1]) / len(woman) * 100, 1))

def q7():
    print("q7")
    man = data[data['sex_categories'] == 1]
    man['Name'] = [name.split(',')[-1].split()[1] for name in man['Name']]
    print(man['Name'].value_counts().head(1))
def q8():
    print("q8")
    man = data[data['sex_categories'] == 1]
    woman = data[data['sex_categories'] == 0]
    print(man[man['Pclass'] == 1].describe()['Age']['mean'] > 40)
    print(woman[woman['Pclass'] == 1].describe()['Age']['mean'] > 40)
    man_gr = man.groupby('Pclass')['Age'].mean()
    woman_gr = woman.groupby('Pclass')['Age'].mean()
    print(man_gr[1] > woman_gr[1] and man_gr[2] > woman_gr[2] and man_gr[3] > woman_gr[3])
    pclasses_greater = data.groupby('Pclass')['Age'].mean()
    print(pclasses_greater[1] > pclasses_greater[2] > pclasses_greater[3])

def main():
    q1()
    printLineSpace()
    q2()
    printLineSpace()
    q3()
    printLineSpace()
    q4()
    printLineSpace()
    q5()
    printLineSpace()
    q6()
    printLineSpace()
    q7()
    printLineSpace()
    q8()

main()