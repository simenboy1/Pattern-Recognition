import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import axes3d

from min_feil_rate import min_feil_rate
from minste_kvadrater import minste_kvadraters
from nnk import nearest_neighbor

def train_test_split(data):
    """ splitte data i trening og test sett

    Treningssettet inneholder odde nummererte data og testsettet inneholder jevne nummererte data

    Argument:
    data => input datasett

    Return:
    Train => Treningssett med odde nummererte data
    Test => Testsett med jevne nummererte data
    """

    #lager en index array med True for hver jevne index og False for hver odde index
    index = np.linspace(1, len(data), len(data))
    boolean = index % 2 == 0

    train = data[np.invert(boolean)]
    test = data[boolean]

    return train, test

def readFile(filename):
    """Konverterer data i filene til matriser ved bruk av pandas-biblioteket

    Argument:
    Filename => navn på tekstfil som skal leses fra

    Return:
    data => Data fra fil som matrise
    """

    data = pd.read_csv(filename, header=None, sep='\s+', engine='python')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    data = data.to_numpy()
    return data

def best_kombinasjon(err_rate, idx_matrise):
    """Finner egenskapskombinasjonen med den minst feilraten

    Argument:
    err_rate => Matrise som inneholder feilraten for hver egenskapskombinasjon
    arr => liste med alle egenskapskombinasjonene


    Return:
    err_rate[best_idx] => minste feilrate
    best_komb => egenskapskombinasjonen med minst feilrate
    """

    best_idx = np.argmin(err_rate)
    best_komb = idx_matrise[best_idx]
    return err_rate[best_idx], best_komb

data = readFile("data\ds-3.txt")

fig = plt.figure()
ax = fig.gca(projection="3d")
class1 = data[data[:, 0] == 1]
class2 = data[data[:, 0] == 2]
surf = ax.scatter(class1[:, 1], class1[:, 2], class1[:, 3], label="Klasse 1")
surf1 = ax.scatter(class2[:, 1], class2[:, 2], class2[:, 3], label="Klasse 2")
ax.set_xlabel("Egenskap 1", size=12)
ax.set_ylabel("Egenskap 2", size=12)
ax.set_zlabel("Egenskap 3", size=12)
ax.set_title("Egenskapsrommet for datasett 2", size=16)
plt.legend(loc="best")
plt.show()

train, test = train_test_split(data)

if train.shape[1] == 5:
    err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4, \
    idx_d1, idx_d2, idx_d3, idx_d4 = nearest_neighbor(test, train)

    error_rate = [err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4]
    index = [idx_d1, idx_d2, idx_d3, idx_d4]

else:
    err_rate_d1, err_rate_d2, err_rate_d3, \
    idx_d1, idx_d2, idx_d3 = nearest_neighbor(test, train)

    error_rate = [err_rate_d1, err_rate_d2, err_rate_d3]
    index = [idx_d1, idx_d2, idx_d3]

for err_rate, idx in zip(error_rate, index):
    NN_err_rate, best_komb = best_kombinasjon(err_rate, idx)
    minste_feil_rate = min_feil_rate(test, train, best_komb)
    min_kvadrater_feil_rate = minste_kvadraters(test, train, best_komb)

    print('-------------------------------------------')
    print(f"Beste kombinasjon: \t {best_komb}")
    print(f"Nearest neighbor: \t {NN_err_rate:.3}")
    print(f"Minste feil rate \t {minste_feil_rate:.3}")
    print(f"Minste kvadraters \t {min_kvadrater_feil_rate:.3}")
    print('-------------------------------------------')

print(err_rate_d1)
print(err_rate_d2)
print(err_rate_d3)
print(err_rate_d4)


