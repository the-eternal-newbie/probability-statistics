# Estadistica
Proyecto de Estadistica y Procesos Estocasticos

## Acerca del programa

Para el desarrollo de las tareas adjuntas en este reporte, se programó una solución computacional en el lenguaje de programación Python, en su versión 3.8.1, la cual se puede conseguir en el siguiente enlace. Para el cómputo matemático de operaciones complejas (como factoriales, exponenciales de ε, etcétera) así como para la creación de gráficas se emplearon librerías matemáticas y de gráficos adicionales a las incluidas en Python. Dichas librerías son NumPy y Matplotlib.
Esta solución computacional cuenta con la capacidad de calcular las medidas de tendencia central y de dispersión, así como las distintas gráficas y distribuciones estadísticas a partir de la lectura de un tipo de archivo llamado JSON, el cual permite una mejor estructura de la información a procesar. Toda la información referente a la configuración y uso del programa y archivos involucrados se describe a continuación.

## ¿Cómo configurar el programa?
Para el correcto funcionamiento del programa de manera local, se necesita tener previamente instalada la versión de Python especificada en el anterior apartado; una vez con dicha versión instalada, se necesita corroborar que también se cuenta con el manejador de paquetes de facto de Python “pip”. Para corroborar dicha instalación, desde la consola en donde se ejecute Python, basta con ingresar el comando `pip –V`.
 
Teniendo el manejador de paquetes de Python, se puede realizar la instalación automática de todos los paquetes requeridos para el programa ejecutando el comando pip install -r requirements.txt estando dentro de la carpeta que contenga el proyecto (por obviedad, una conexión a internet es requerida para la descarga de paquetes)
 

## ¿Cómo usar el programa?

Una vez instalados los paquetes necesarios, el programa puede ser ejecutado desde la misma consola corriendo el comando que ejecute a Python (dicho comando es configurable y puede ser tanto “python”, “python3” o simplemente “py”, dependiendo de la instalación realizada en el sistema) acompañado del nombre del archivo (script.py). Para el caso personal, el comando para correr el programa es `py script.py`. Habiendo hecho esto, se mostrará en pantalla un menú con las opciones del programa.

## Archivos
Para el cálculo de los datos se emplea un tipo de archivo llamado JSON, el cual respeta una sintaxis específica, la cual consta de una relación clave-valor, es decir, si tenemos una clave con nombre “enero”, se le asigna un valor de 25, pero a esta también se le puede añadir más datos como se muestra a continuación.

Esta estructura del archivo obedece a lo que en programación se conoce como “diccionario”. De esta manera es más fácil identificar los datos que corresponden a cierto estado (en este caso), como, por ejemplo, en el caso de los tiempos de traslado, se puede hacer un archivo cuyas claves sean los meses y estos contengan días y los valores de los días hagan referencia al tiempo en horas y minutos que duró el trayecto.
 
Todo esto permite una mejor manipulación de los datos y beneficia en el caso de querer modificar los datos o crear nueva información si se quisiera adaptar el enfoque del programa más allá de los tiempos de traslado solamente.
El proyecto también incluye una carpeta de imágenes que se generan cuando se solicita visualizar alguna gráfica, además de otros archivos que sirven para hacer los cálculos estadísticos pertinentes.
