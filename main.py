import math

import PySimpleGUI as simpleGui
import matplotlib.pyplot as plt
import numpy as np
import sympy as sympy
from sympy import *

#globalne mu?
def func(x):
    return eval(values["-WZOR-"])

def funcCurry(fun, x):
    # TU WPISUJE 3 ROZNE FUNCJE KARY
    # KOLO (O SRODKU W PUNKCIE (2,3) i PROMIENIU 1)
    # f = log(-1*(x[0]-2)**2-1*(x[1]-3)**2+1)
    # KWADRAT (O SRODKU W PUNKCIE (2, 3) I BOKU DDLUGOSCI 2)
    # g = log(x[0]-1) + log(3-x[0]) + log(x[1]-4) + log(2-x[1])
    # POLKOLE
    # h = log(-0.5*(x[0]-2)-0.5*(x[1]-2)+2) + log(2-x[1])
    f = log(-1 * (x[0] - 2) ** 2 - 1 * (x[1] - 3) ** 2 + 1)
    return fun - f

def funcWykres(x):
    return eval(values["-WZOR-"])

def stopDokladnosc(x0, x_new, i):
    suma_roznic_do_kwadratu = 0
    for element_number in range(x0.size - 1):
        suma_roznic_do_kwadratu += (x0.item(element_number) - x_new.item(element_number)) ** 2

    distance = math.sqrt(suma_roznic_do_kwadratu)

    stop_value = 0.1
    return distance <= stop_value


def stopIteracje(x0, x_new, i):
    stop_value = float(values["-STOP-WARTOSC-"])
    return i >= stop_value


def getGrad(x0):
    f_variables = list(sympy.symbols(' '.join('x%d' % i for i in range(x0.size))))
    grad_fx = []
    subs = assign_values_in_point(f_variables, x0)
    for variable in f_variables:
        grad_fx.append(diff(funcCurry(func, f_variables), variable).evalf(subs=subs))
    return grad_fx


def assign_values_in_point(f_variables, x0):
    subs = {}
    for element in range(x0.size):
        subs[f_variables[element]] = x0.item(element)
    return subs


def rysujWykres(punkty):

    x_interval = (float(values["-X1-"]), float(values["-Y1-"]))
    y_interval = (float(values["-X2-"]), float(values["-Y2-"]))

    x_points = np.linspace(x_interval[0], x_interval[1], 100)
    y_points = np.linspace(y_interval[0], y_interval[1], 100)
    X, Y = np.meshgrid(x_points, y_points)

    func3d_vectorized = []
    for point_x in x_points:
        row = []
        for point_y in y_points:
            row.append(funcWykres([point_y, point_x]))
        func3d_vectorized.append(row)
        # evaluation of the function on the grid
    Z = func3d_vectorized

    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')

    punkty_as_arrays = []
    for punkt in punkty:
        punkty_as_arrays.append(np.squeeze(np.asarray(punkt)))
    x, y = zip(*punkty_as_arrays)
    plt.scatter(x, y, c="r")
    plt.text(x[len(x) - 1], y[len(y) - 1], "Xi", c="w")
    plt.text(x[0], y[0], "X0", c="grey")
    plt.title("f(x,y) = " + values["-WZOR-"])
    plt.savefig("Wykres.png")
    plt.ginput(4)
    plt.show()


def program(x_start, warunek_stopu):
    punkty = [x_start]
    h0 = np.eye(x_start.size)
    i = 0
    learning_rate = 0.1

    while true:
        # pobierz punkt
        x0 = punkty[i]
        # obliczenie gradientu w punkcie
        grad_f_x0 = getGrad(x0)
        # kierunek spadku
        d = np.multiply(learning_rate, np.dot(h0, grad_f_x0))

        # wyznacz kolejny punkt
        x_new = np.subtract(x0, d)
        punkty.append(x_new)

        # DFP
        grad_f_x_new = getGrad(x_new)
        q = np.subtract(grad_f_x_new, grad_f_x0)  # y
        p = np.subtract(x_new, x0)  # k
        h0 = h0 + p.reshape([x0.size, 1]) * p / np.dot(p, q) \
            - np.dot(h0, q).reshape([x0.size, 1]) * np.dot(q, h0) / np.dot(np.dot(q, h0), q)

        i += 1

        if stopDokladnosc(x0, x_new, i):
            print(punkty)
            if x0.size == 2:
                rysujWykres(punkty)
            break
    print('i: ')
    print(i)

mu = 1.0

#Tu jest ta pętla, po której ma być iterowany program, na razie na sztywno, jak nam zostanie czas, to wtedy można zrobić, ze sprawdzeniem, czy coś się zmienilo w tablicy
#while mu > 0.01:
#    program(x_start, stopDokladnosc(x0, x_new, i))
#    mu = mu/1.5

# Układ okna
layout = [
    [simpleGui.Text("Funkcja:")],
    [simpleGui.InputText("x[0]**2+cos(x[1])", key="-WZOR-")],

    [simpleGui.Text("Punkt początkowy (oddziel spacjami):")],
    [simpleGui.InputText("9 8", key="-X0-", size=(5, 1))],

    [simpleGui.Text("Kryterium stopu:")],
    [simpleGui.Listbox(values=["Liczba iteracji", "Dokładność"], key="-STOP-", size=(20, 2))],
    [simpleGui.InputText("9", key="-STOP-WARTOSC-", size=(5, 1))],

    [simpleGui.HSeparator()],
    [simpleGui.Text("Zakresy wykresu dla funkcji 2 zmiennych:")],

    [simpleGui.Text("Wartości X:")],
    [simpleGui.InputText("-10", key="-X1-", size=(5, 1))],
    [simpleGui.Text("do:")],
    [simpleGui.InputText("10", key="-Y1-", size=(5, 1))],

    [simpleGui.Text("Wartości Y:")],
    [simpleGui.InputText("-10", key="-X2-", size=(5, 1))],
    [simpleGui.Text("do:")],
    [simpleGui.InputText("10", key="-Y2-", size=(5, 1))],

    [simpleGui.Text(key="-KOMUNIKAT-", size=(30, 1))],
    [simpleGui.Button("Szukaj minimum", key="-OK-")]
]
window = simpleGui.Window(
    "Matplotlib Single Graph",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="left",
    font="Helvetica 14",
)

while True:
    event, values = window.read()
    if event == 'Exit' or event == simpleGui.WIN_CLOSED:
        break
    if event == "-OK-":
        wzor = values["-WZOR-"]
        x_start = values["-X0-"].split()
        x_start = [float(i) for i in x_start]
        if values["-STOP-"][0] == "Liczba iteracji":
            warunekStopu = stopIteracje
        else:
            warunekStopu = stopDokladnosc
        program(np.matrix(x_start, dtype=float), warunekStopu)