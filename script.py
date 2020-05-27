# paquetes auxiliares para la graficación de diagramas
# y operaciones matemáticas
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

#######################################################################################
#                                                                                     #
#                 Programa de estadistica y procesos estocásticos                     #
#                                                                                     #
#                                                                                     #
#   El presente programa cubre los distintos cálculos y proyecciones gráficas de      #
#   los temas abarcados en el curso de estadistica y procesos estocásticos, que       #
#   engloban las medidas de tendencia central, medidas de distribución y gráficas     #
#   como la circular, de caja y bigotes, Pareto, histograma, polígono de frecuencia   #
#   y de dispersión. La configuración del programa y el uso del mismo se explica      #
#   a detalle en el reporte adjunto al programa.                                      #
#                                                                                     #
#                                                                                     #
#                                                                                     #
#                                            Desarrollado @Carlos Adonis Vara Pérez   #
#                                            Código: 216787671                        #
#                                            Contacto: ca.varaperez@gmail.com         #
#                                                                                     #
#######################################################################################


# Clase auxiliar contiene métodos para transformar datos y hacer cálculos globales
# sobre el conjunto de datos
class transformers(object):
    # Transforma el diccionario de datos (que contiene los datos agrupados por mes
    # y día) en una lista simple con todos los datos consecutivos (tiempos)
    @staticmethod
    def to_list(data, transform=False):
        data_list = []
        # itera por cada mes
        for month in data:
            # itera por cada día
            for day in data[month]:
                if(transform):
                    data_list.append(transformers.to_minutes(data[month][day]))
                else:
                    data_list.append(data[month][day])
        return(data_list)

    # Transforma los datos (texto) en un formato de minutos para poder hacer
    # los cálculos correspondientes
    @staticmethod
    def to_minutes(date_string):
        return((int(date_string[:2]) * 60) + int(date_string[3:]))

    # Al contrario de la función anterior, transforma los minutos a una
    # cadena horaria
    @staticmethod
    def to_hours(time):
        hours = int(time / 60)
        minutes = int(time % 60)
        if(minutes < 10):
            minutes = '0' + str(minutes)
        if (hours < 10):
            hours = '0' + str(hours)

        return(str("{}:{}".format(hours, minutes)))

    # Calcula el la suma total de los datos y la cantidad de datos
    # existente en el diccionario de datos
    @staticmethod
    def total_amount(data):
        amount = 0
        total = 0
        for month in data:
            for day in data[month]:
                amount += 1
                total += transformers.to_minutes(data[month][day])
        return(total, amount)

    # Calcula la variación del conjunto de datos
    @staticmethod
    def total_variation(data, mean):
        total = 0
        for month in data:
            for day in data[month]:
                total += pow(
                    (transformers.to_minutes(data[month][day]) - mean), 2)

        return(total)

    # Calcula el porcentaje correspondiente a cada valor en el conjunto
    @staticmethod
    def percentages(data):
        # cuenta la ocurrencia de datos
        values = {i: data.count(i) for i in data}
        amount = len(data)
        # por cada dato único le asigna el porcentaje correspondiente del conjunto
        for x in values:
            percentage = (values[x] / amount) * 100
            values[x] = round(percentage, 2)
        return(values)

    # Retorna las horas de una cadena horaria (útil para realizar la correlación
    # de datos con respecto a las horas de salida)
    @staticmethod
    def only_hours(data):
        return(int(data[:2]))

    # Genera dos listas (new_departures, new_durations) las cuales guardan
    # el orden de referencia que tienen; es decir, ordena una lista con respecto
    # a otra dependiendo el día
    @staticmethod
    def join_correlation(data1, data2):
        new_durations = []
        temp_dict_1 = {}
        temp_dict_2 = {}
        i = 0
        for departure, duration in zip(data1, data2):
            temp_dict_1[i] = departure
            temp_dict_2[i] = duration
            i += 1

        sorted_data = {k: v for k, v in sorted(
            temp_dict_1.items(), key=lambda item: item[1])}

        new_departures = list(sorted_data.values())
        index_list = list(sorted_data.keys())
        for i in index_list:
            new_durations.append(temp_dict_2[i])

        return(new_departures, new_durations)

# Medidas de tendencia central (mdtc)


class mdtc(object):
    """
    media aritmética
        @param(total): la suma total del conjunto de datos (float)
        @param(amount): la cantidad del conjunto de datos (list)
    """
    @staticmethod
    def mean(total, amount):
        mean_data = total / amount
        return(round(mean_data, 2))

    """
    mediana
        @param(total): la suma total del conjunto de datos (list)
        @param(amount): la cantidad del conjunto de datos (int)
    """
    @staticmethod
    def median(data, amount):
        data.sort()
        if(amount % 2 == 2):
            factor = int(amount / 2)
            return((data[factor] + data[factor+1]) / 2)
        else:
            factor = round(amount / 2)
            return(data[factor])

    """
    media aritmética
        @param(data): la lista de los valores del conjunto de datos (list)
    """
    @staticmethod
    def mode(data):
        # retorna el valor máximo de un par ordenado que hace referencia
        # al dato y su repetición en el conjunto
        return(max(set(data), key=data.count))

# Medidas de dispersión


class mdd(object):
    """
    rango
        @param(data): la lista de los valores del conjunto de datos (list)
    """
    @staticmethod
    def range(data):
        # ordena los datos y resta el último menos el primero
        data.sort()
        return(data[-1] - data[0])

    """
    desviación media
        @param(data): el diccionario del conjunto de datos (dict)
        @param(amount): la cantidad de valores en el conjunto de datos (int)
        @param(mean): el promedio del conjunto de datos (float)
    """
    @staticmethod
    def mad(data, amount, mean):
        total = 0
        for month in data:
            # sumatoria de los datos de x_i hasta x_n menos el promedio del conjunto
            for day in data[month]:
                total += (transformers.to_minutes(data[month][day]) - mean)
        # dividido entre el número total de datos N
        return(total / amount)

    """
    covarianza
        @param(data1): el diccionario del conjunto de datos uno [tiempos de traslado] (dict)
        @param(data2): el diccionario del conjunto de datos dos [horas de salida] (dict)
    """
    @staticmethod
    def covariance(data1, data2):
        total = 0
        mean_1 = np.mean(data1)
        mean_2 = np.mean(data2)
        # la sumatoria del producto de los valores de x_i hasta x_n menos el promedio del conjunto X
        # por los valores de y_i hasta y_n menos el promedio del conjunto Y
        for x, y in zip(data1, data2):
            total += ((x - mean_1) * (y - mean_2))
        # dividido entre el número total de datos N
        return(round((total/len(data1)), 4))

    """
    desviación estándar
        @param(variation): el valor de la varianza del conjunto de datos (float)
    """
    @staticmethod
    def standar_dev(variation):
        # la desviación estándar no es más que la raíz cuadrada de la varianza
        return(math.sqrt(variation))

    """
    varianza
        @param(total): el valor de la sumatoria dada por la función total_variation (float)
        @param(amount): la cantidad de datos del conjunto (int)
    """
    @staticmethod
    def variation(total, amount):
        return(total / (amount))

    """
    coeficiente de variación
        @param(sigma): el valor de la desviación estándar (float)
        @param(mean): el valor promedio del conjunto de datos (float)
    """
    @staticmethod
    def var_coefficient(sigma, mean):
        return(sigma / mean)

    """
    coeficiente de correlación
        @param(data1): el conjunto de datos uno [tiempos de traslado] (list)
        @param(data2): el conjunto de datos dos [horas de salida] (list)
        @param(covariance): el valor de la covarianza (float)
    """
    @staticmethod
    def correl_coefficient(data1, data2, covariance):
        std_1 = np.std(data1)
        std_2 = np.std(data2)
        # el valor de la covarianza entre el producto de las desviaciones estándar
        # de ambos conjuntos X y Y
        return(round(((covariance) / (std_1 * std_2)), 4))


# Diagramas y gráficos
class diagrams(object):
    """
    diagrama de caja y bigotes
    * para cuestiones demostrativos, se ingresan datos que pudieran ser atípicos
        @param(data): el conjunto de datos (list)
        @param(atypic_max, atypic_min): el conjunto de datos atipicos [horas de salida] (list)
    """
    @staticmethod
    def boxplot(data, atypic_max, atypic_min):
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_title('Diagrama de caja representando los tiempos de traslado')
        plt.ylabel('Tiempo de traslado (min.)')

        # se calculan los valores de la media y mediana para mostrar en el diagrama
        ax1.text(1.1, np.mean(data), r'media', fontsize=10)
        ax1.text(1.1, np.median(data), r'mediana', fontsize=10)

        atypic_label_1 = 'Datos atípicos {}'.format(atypic_max)
        atypic_label_2 = 'Datos atípicos {}'.format(atypic_min)

        ax1.text(1.1, 5, atypic_label_1, fontsize=8)
        ax1.text(1.1, 110, atypic_label_2, fontsize=8)
        final_data = np.concatenate((data, atypic_max, atypic_min))

        ax1.boxplot(final_data, labels=['Enero-Marzo'],
                    showmeans=True, meanline=True)
        plt.savefig('img/caja.png')
        plt.show()

    """
    diagrama circular
        @param(data): el conjunto de datos (list)
    """
    @staticmethod
    def pie_chart(data):
        # se calculan los porcentajes de los valores del conjunto
        # para posteriormente asignarlos a cada porción de la gráfica
        values = transformers.percentages(data)
        labels = []
        for x in values.keys():
            labels.append(str(x) + " min.")
        explode = np.full(len(values.keys()), 0.05)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Diagrama circular representando los tiempos de traslado')
        ax.pie(values.values(), autopct='%.1f%%',
               startangle=90, pctdistance=0.85, explode=explode)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        ax.axis('equal')
        ax.legend(labels, loc="lower left")
        plt.tight_layout()
        plt.savefig('img/circular.png')
        plt.show()

    """
    diagrama de Pareto
        @param(data): el conjunto de datos (list)
    """
    @staticmethod
    def pareto(data):
        # se obtiene la frecuencia de los valores del conjunto y se ordenan
        # de manera decreciente
        values = {i: data.count(i) for i in data}
        new_values = {k: v for k, v in sorted(
            values.items(), reverse=True, key=lambda item: item[1])}

        string_values = []
        for x in new_values.keys():
            string_values.append(str(x))

        df = pd.DataFrame(
            {'minutes': list(new_values.values())})
        df.index = string_values
        df = df.sort_values(by='minutes', ascending=False)
        df["cumpercentage"] = df["minutes"].cumsum()/df["minutes"].sum()*100

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.ylabel('Frecuencia')
        plt.xlabel('Tiempo de traslado (min.)')

        ax.set_title(
            'Diagrama de Pareto representando los tiempos de traslado')
        ax.bar(df.index, df["minutes"], color="C0")

        ax2 = ax.twinx()
        ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
        ax2.yaxis.set_major_formatter(PercentFormatter())

        ax.tick_params(axis="y", colors="C0")
        ax2.tick_params(axis="y", colors="C1")
        plt.savefig('img/pareto.png')
        plt.show()

    """
    diagrama de dispersión
        @param(data1): el conjunto de datos uno [tiempos de traslado] (list)
        @param(data2): el conjunto de datos dos [horas de salida] (list)
    """
    @staticmethod
    def scatter_plot(data1, data2):
        # se realiza la correlación de datos
        x, y = transformers.join_correlation(data1, data2)

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.ylabel('Tiempo de traslado (min.)')
        plt.xlabel('Hora de salida')

        ax.set_title(
            'Diagrama de Correlacion entre los tiempos de traslado y las horas de salida')
        ax.plot(x, y, 'x', color='red')
        new_data = []
        for x in data1:
            new_data.append(transformers.only_hours(x))
        # se calcula la covarianza y su coeficiente de relación
        covariance = mdd.covariance(new_data, data2)
        coeff = mdd.correl_coefficient(new_data, data2, covariance)
        text = 'Covarianza: {}\nCoeficiente de correlacion: {}'.format(
            covariance, coeff)
        ax.text(12, 35, text, style='normal',
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        plt.savefig('img/dispersion.png')
        plt.show()

    """
    histograma y polígono de frecuencias
        @param(data): el conjunto de datos (list)
        @param(mu): el valor de la desviación media (float)
        @param(sigma): el valor de la desviación estándar (float)
    """
    @staticmethod
    def histogram(data, mu, sigma):
        fig, ax = plt.subplots(figsize=(12, 6))

        # the histogram of the data
        n, bins, patches = ax.hist(
            data, alpha=0.5, stacked=True, color='red')

        ax.plot(bins[:-1]+2.5, n, '-o', color='red', alpha=1.5)
        # add a 'best fit' line
        ax.set_xlabel('Tiempo de traslado (min)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma de los tiempos de traslado')

        sigma_label = 'Desviacion estándar: {}'.format(round(sigma, 4))
        mu_label = 'Desviacion media: {}'.format(round(mu, 4))
        ax.text(71.5, 20, sigma_label, fontsize=10)
        ax.text(71.5, 19, mu_label, fontsize=10)
        red_patch = mpatches.Patch(
            color='red', label='Polígono de frecuencias')
        ax.legend(handles=[red_patch])
        fig.tight_layout()
        plt.savefig('img/histograma.png')
        plt.show()

    """
    tabla de datos
    * esta función sirve para desplegar los datos almacenados en el archivo
    * de los tiempos de traslado
        @param(data_dict): el diccionario del conjunto de datos (dict)
    """
    @staticmethod
    def tables(data_dict):
        for month in data_dict:
            print('Mes - {}'.format(month))
            i = 0
            string = ''
            month_date = data_dict[month]
            last = list(month_date.keys())[-1]
            for day in month_date:
                if(i <= 2):
                    string += 'Dia {} - Tiempo de traslado: {}\t'.format(day,
                                                                         data_dict[month][day])
                else:
                    print(string + 'Dia {} - Tiempo de traslado: {}\t'.format(day,
                                                                              data_dict[month][day]))
                    string = ''
                    i = -1
                i += 1
                if(day == last):
                    print(string)
            print('\n')

    @staticmethod
    def binomial(data, n):
        x = []
        y = []
        for x2 in range(n):
            y.append(distributions.binomial_dist(data, n, x2+1) * 100)
            x.append(x2)
        fig, ax = plt.subplots(figsize=(12, 6))
        cm = plt.cm.get_cmap('YlGn')
        colors = cm(y)
        ax.bar(x, y, color=colors)
        ax.set_xlabel(
            'Dias en los que se llegue temprano dentro de un lapso de {} días'.format(n))
        plt.xticks(np.arange(min(x), max(x), 5.0))
        ax.set_ylabel('Probabilidad de que suceda')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title(
            'Distribución binomial de la probabilidad de días en los que se llegue temprano')
        fig.tight_layout()
        plt.savefig('img/binomial.png')
        plt.show()

    @staticmethod
    def poisson(data, n, top, state):
        x = []
        y = []
        for x2 in range(top):
            y.append(distributions.poisson_dist(data, n, x2+1) * 100)
            x.append(x2)
        fig, ax = plt.subplots(figsize=(12, 6))
        cm = plt.cm.get_cmap('YlOrRd')
        colors = cm(y)
        ax.bar(x, y, color=colors)
        ax.set_xlabel(
            'Cantidad de personas contagiadas dentro de una muestra de {} personas'.format(n))
        plt.xticks(np.arange(min(x), max(x), 5.0))
        ax.set_ylabel('Probabilidad de contagio')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title(
            'Distribución de Poisson de la probabilidad de personas contagiadas por COVID-19 en {}'.format(state))
        fig.tight_layout()
        plt.savefig('img/poisson.png')
        plt.show()


class distributions():
    @staticmethod
    def binomial_comb(n, x):
        return(math.factorial(n) / (math.factorial(x) * math.factorial((n - x))))

    @staticmethod
    def binomial_dist(data, n2, x2):
        # Se calcula el porcentaje de éxitos en la muestra
        n = len(data)
        # Cantidad total de resultados exitosos (días en los que se llegó temprano)
        x = {i: data.count(i) for i in data}[True]
        # Para la distribución binomial, se calcula el combinatorio de los datos que
        # se quieren saber
        n_2_x = distributions.binomial_comb(n2, x2)
        # los porcentajes de éxito y fracaso se calculan a partir de n y x
        p = round((x / n), 2)
        q = round((1 - p), 2)
        # mientras que la distribución se calcula tomando los valores que
        # se desean calcular a partir de n2 y x2
        return(round((n_2_x * math.pow(p, x2) * math.pow(q, (n2-x2))), 4))

    @staticmethod
    def poisson_dist(data, n, x):
        delta = (data['contagios'] / data['habitantes']) * n
        return((np.exp(-delta) * (math.pow(delta, x))) / math.factorial(x))


# limpiar pantalla dependiendo el s.o.
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    # Archivo de datos uno (tiempos de traslado)
    file = open("duracion.json", "r")
    json_file = str(file.read())
    file.close()
    data = json.loads(json_file)

    # Archivo de datos dos (horas de salida)
    file = open("salidas.json", "r")
    json_file = str(file.read())
    file.close()
    data2 = json.loads(json_file)

    file = open("llegadas.json", "r")
    json_file = str(file.read())
    file.close()
    data3 = json.loads(json_file)
    data_list3 = transformers.to_list(data3)

    file = open("covid.json", "r")
    json_file = str(file.read())
    file.close()
    data4 = json.loads(json_file)

    total, amount = transformers.total_amount(data)
    data_list = transformers.to_list(data, True)
    data_list2 = transformers.to_list(data2)

    # calcula todos los valores necesarios para el despliegue
    # de información estadística y la creación de diagramas
    mean = mdtc.mean(total, amount)
    total_variation = transformers.total_variation(data, mean)
    median = mdtc.median(data_list, amount)
    mode = mdtc.mode(data_list)
    data_range = mdd.range(data_list)
    variation = mdd.variation(total_variation, amount)
    sigma = mdd.standar_dev(variation)
    mad = mdd.mad(data, amount, mean)
    var_coefficient = mdd.var_coefficient(sigma, mean)

    # Menú de opciones
    opt = -1
    while(opt != 7):
        cls()
        print('\t\tB I E N V E N I D O\n\n')
        print('\t\t0. Consultar datos')
        print('\t\t1. Ver resultados estadísticos\n\t\t2. Ver diagrama de caja')
        print('\t\t3. Ver diagrama circular\n\t\t4. Ver de Pareto')
        print('\t\t5. Ver diagrama de dispersion\n\t\t6. Ver histograma')
        print('\t\t7. Ver distribución binomial\n\t\t8. Ver distribución de Poisson (COVID-19)')
        print('\t\t9. Salir\n')
        while(opt != '0' and opt != '1' and opt != '2' and opt != '3' and opt != '4' and opt != '5' and opt != '6' and opt != '7' and opt != '8' and opt != '9'):
            opt = str(input('Seleccione una opcion valida: '))

        if(opt == '0'):
            cls()
            diagrams.tables(data)
        elif(opt == '1'):
            cls()
            print("""TOTAL DE DATOS: {}\n\n\t\tMedidas de tendencia central\n\n\t\tMedia: {}\n\t\tMediana: {}\n\t\tModa: {}\n\n
    \t\tMedidas de dispersión\n\n\t\tRango: {}\n\t\tDesviación media: {}\n\t\tDesviación estándar: {}
\t\tVarianza: {}\n\t\tCoeficiente de variación: {}\n""".format(amount,
                                                               mean, median, mode, data_range, round(mad, 4), round(
                                                                   sigma, 4), round(variation, 4), round(var_coefficient, 4)
                                                               ))
        elif(opt == '2'):
            cls()
            diagrams.boxplot(data_list, [5, 6, 9], [110, 109, 115])
        elif(opt == '3'):
            cls()
            diagrams.pie_chart(data_list)
        elif(opt == '4'):
            cls()
            diagrams.pareto(data_list)
        elif(opt == '5'):
            cls()
            diagrams.scatter_plot(data_list2, data_list)
        elif(opt == '6'):
            cls()
            diagrams.histogram(data_list, mad, sigma)
        elif(opt == '7'):
            cls()
            n = int(input('Ingrese la cantidad de la muestra: '))
            diagrams.binomial(data_list3, n)
        elif(opt == '8'):
            cls()
            keys = list(data4.keys())
            i = 1
            for key in keys:
                print('{}. {}'.format(i, key))
                i += 1
            opt = int(
                input('\nSeleccione un estado para realizar la distribución: '))
            city = keys[opt-1]
            city_data = data4[city]
            diagrams.poisson(city_data, 50000, 100, city)

        if(opt != '9'):
            input('\nPresione enter para regresar al menú anterior: ')
            opt = -1
        elif(opt == '9'):
            exit()
