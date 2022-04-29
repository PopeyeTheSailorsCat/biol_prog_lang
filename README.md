# Реализация численного отжига.
Данный репозиторий содержит в себе пример реализации алгоритма эмпирической оптимизации: численный отжиг.
Реализация состоит из нескольких модулей, имеет логирование и обработку ошибок.
Работа программы легко настраивается с помощью config.py:
* STARTING_TEMPERATURE - определяет начальную температуру отжига
* STOPPING_TEMPERATURE - условие остановки отжига
* ITERATION_PER_TEMPERATURE - сколько раз мы рассматриваем соседей за текущую температуру.

* CLOSEST_STEP - ближайшее растояние в процентах от интервала, которое мы можем пройти
* FURTHER_STEP - наибольшее растояние в процентах от интервала, которое мы можем пройти

SAVING_DIVISION_EPSILON - для избежания деления на 0 при подсчете вероятности.


* COOLING_SCHEDULE - определяет формат изменения T. Есть на выбор:
  * linear - линейный режим
  * geom - геометрический режим
  * log_1 - логарифмический режим
  * log_2 - режим коши

* linear_param - значение параметра в  линейном режиме
* geom_param - значения параметра в геометрическом режиме

* CHECKPOINT_FILE - Путь к файлу, куда записывать состояния во время исполнения
* CHECKPOINT_EVERY_COOLING - как часто относительно изменений температуры сохранять состояние

* RESULT_FILE - дефолтный файл сохранения

* DEFAULT_EXAMPLE_PATH - пример который запустится, если нет пути к Input.


# Входные данные:
Файл с исходными данными представляет из себя файл, где
* в первой строчке в питоновоском формате описывается функция, которая будет минимизирвоаться.
* во второй строчке начальные значения переменных.
* в последующих строчках название переменной, минимальное значение, максимальное значение
![alt text](imgs/input_example.PNG)
# Запуск программы:
![alt text](imgs/example_h.PNG)

Таким образом самый простой вызов программы будет
```
python main.py
```
Который запустит рассчет по дефолтному инпуту.

Можно рассмотреть примерные варианты, которые дадут информацию о минимуме значений функции:
````commandline
python main.py -e 1
````
Запуск на вашем inpute c outputОм будет выглядеть как
```commandline
python main.py -p input_path -o output_path
```

Запуск с какого то checkpoint-а выглядит так:
```commandline
python main.py -p input_path  -c -ch_pth checkpoint_path
```

