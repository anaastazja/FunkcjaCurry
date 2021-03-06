import math

import PySimpleGUI as simpleGui
import matplotlib.pyplot as plt
import numpy as np
import sympy as sympy
from matplotlib import patches
from sympy import *


def func(x, x_values, mu, funcOgraniczenia):
    return eval(values["-WZOR-"]) - funcOgraniczenia(x_values) * mu


def funcKolo(x):
    # KOLO (O SRODKU W PUNKCIE (2,3) i PROMIENIU 1)
    x_ = -1 * (x[0] - 2) ** 2 - (x[1] - 3) ** 2 + 25
    return math.log(x_, 10)


def funcKwadrat(x):
    # KWADRAT (O SRODKU W PUNKCIE (2, 3) I BOKU DDLUGOSCI 10)
    return math.log(x[0] + 3) + math.log(7 - x[0]) + math.log(x[1] + 2) + math.log(8 - x[1])


def funcPolkole(x):
    return math.log(-1* (x[0] - 2) ** 2 -(x[1] + 1) ** 2 + 25) + math.log(1 + x[1], 10)


def funcWykres(x):
    return eval(values["-WZOR-"])


def stopDokladnosc(x0, x_new, i):
    suma_roznic_do_kwadratu = 0
    for element_number in range(x0.size - 1):
        suma_roznic_do_kwadratu += (x0.item(element_number) - x_new.item(element_number)) ** 2

    distance = math.sqrt(suma_roznic_do_kwadratu)

    stop_value = 0.01
    return distance <= stop_value


def getGrad(x0, mu, funcOgraniczenia):
    f_variables = list(sympy.symbols(' '.join('x%d' % i for i in range(x0.size))))
    grad_fx = []
    subs = assign_values_in_point(f_variables, x0)
    for variable in f_variables:
        grad_fx.append(
            diff(func(f_variables, [x0.item(0), x0.item(1)], mu, funcOgraniczenia), variable).evalf(subs=subs))
    return grad_fx


def assign_values_in_point(f_variables, x0):
    subs = {}
    for element in range(x0.size):
        subs[f_variables[element]] = x0.item(element)
    return subs


def rysujWykres(punkty):

    a = -7
    b = 10

    x_points = np.linspace(a, b, 100)
    y_points = np.linspace(a, b, 100)
    X, Y = np.meshgrid(x_points, y_points)

    if values["-STOP-"][0] == "Koło":
        circle = plt.Circle((2, 3), 5, color="r", fill=False)
        plt.gca().add_patch(circle)
    if values["-STOP-"][0] == "Kwadrat":
        rectangle = plt.Rectangle((-3, -2), 10, 10, edgecolor='r', fill=None)
        plt.gca().add_patch(rectangle)
    if values["-STOP-"][0] == "Półkole":
        semicircle = patches.Arc((2, -1), 10, 10, angle=0.0, theta1=0.0, theta2=180.0, color='r')
        plt.gca().add_patch(semicircle)
        plt.hlines(y=-1, xmin=-3, xmax=7, color='r')

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
    plt.show()


def program(x_start, funcOgraniczenia):
    punkty = [x_start]
    h0 = np.eye(x_start.size)
    i = 0
    learning_rate = 0.2
    mu = 1.0
    work = True

    while work:
        # pobierz punkt
        x0 = punkty[i]
        # obliczenie gradientu w punkcie
        try:
            grad_f_x0 = getGrad(x0, mu, funcOgraniczenia)
            # kierunek spadku
            d = np.multiply(learning_rate, np.dot(h0, grad_f_x0))

            # wyznacz kolejny punkt
            x_new = np.subtract(x0, d)
            punkty.append(x_new)

            print("mu ", mu)

            # DFP
            grad_f_x_new = getGrad(x_new, mu, funcOgraniczenia)
            q = np.subtract(grad_f_x_new, grad_f_x0)  # y
            p = np.subtract(x_new, x0)  # k
            h0 = h0 + p.reshape([x0.size, 1]) * p / np.dot(p, q) \
            - np.dot(h0, q).reshape([x0.size, 1]) * np.dot(q, h0) / np.dot(np.dot(q, h0), q)

            i += 1

            if stopDokladnosc(x0, x_new, i):
                print(punkty)
                if x0.size == 2:
                    rysujWykres(punkty)
                    work = False
        except ValueError as e:
            print(e)
            window["-KOMUNIKAT-"].update('Sprawdź na wykresie czy punkt x0 na pewno znajduję się wewnątrz ograniczeń')
            rysujWykres(punkty)
            work = False

        mu = mu / 1.5
    print('i: ')
    print(i)


# Układ okna
layout = [
    [simpleGui.Text("Funkcja:")],
    [simpleGui.InputText("(x[0]-2)**2+(x[1]-3)**2", key="-WZOR-")],

    [simpleGui.Text("Punkt początkowy (oddziel spacjami):")],
    [simpleGui.InputText("1.5 2.5", key="-X0-", size=(5, 1))],

    [simpleGui.Text("Obszar ograniczający:")],
    [simpleGui.Listbox(values=["Koło", "Półkole", "Kwadrat"], key="-STOP-", size=(20, 3))],

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
        try:
            if values["-STOP-"][0] == "Koło":
                funcOgraniczenia = funcKolo
            elif values["-STOP-"][0] == "Półkole":
                funcOgraniczenia = funcPolkole
            else:
                funcOgraniczenia = funcKwadrat
            program(np.matrix(x_start, dtype=float), funcOgraniczenia)
        except:
            window["-KOMUNIKAT-"].update('Sprawdź wpisane dane')
