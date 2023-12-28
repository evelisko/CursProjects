Чат-бот для телеграм.

1. Реализовать чат бот. +
1.1. Загрузить модель на диск. И, разворачивать ее локально из файла.
4. Поискать датасет для дообучения модели.
   Обучить модель.
2. Развернуть проект в docker. +
3. Запустить LLM-модель для работы с чат-ботом в docker.

5. Реализовать многозадачный чат-бот. с помощью использования нескольких адаптеров.
6. Вынести праметры модели в файл конфигураций.
7. Если будет время. Можно еще добавить проверку на токсичность текста. И, отвечать только в случае, если текст не токсичный.
-------------------------------------------------------
Необходимо так-же реализовать логирование 
+ мониторинг потребления ресурсов.
-------------------------------------------------------

load loara adapter - загрузка исходной модели из higgingFace.  
test lora adapter - проверка загрузки адаптера загруженного из файла.
train lora adapter - тренировка адаптера.


train config. 

Подготовить датасет для дообучения модели.
привести датает к стандартному формату.


https://habr.com/ru/articles/766096/
https://habr.com/ru/companies/neoflex/articles/722584/


Имя чат-бота.
Токен для бота.

----------------------------------------------------------------------------------------------
Надо оставить только decker-compose. Остальные скрипты выкинуть.
Сохранить модель на диск с возможностью ее локальной загрузки.

Установить  NVIDIA Container Toolkit согласно инструкции - 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
Чтобы установить драйверы NVIDIA Docker:
sudo apt install nvidia-docker2
Чтобы изменения вступили в силу, перезагрузите компьютер с помощью следующей команды:
 sudo reboot
 Проверка доступности графического процессора NVIDIA из контейнеров Docker в Ubuntu 22.04 LTS:

 # Running an interactive CUDA session isolating the first GPU
docker run -ti --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 nvidia/cuda

# Querying the CUDA 7.5 compiler version
docker run --rm --runtime=nvidia nvidia/cuda:7.5-devel nvcc --version

# Хорошая статейка на эту тему.
https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/


# DEPRECATION NOTICE

This project has been superseded by the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).

The tooling provided by this repository has been deprecated and the repository archived.

The `nvidia-docker` wrapper is no longer supported, and the NVIDIA Container Toolkit has been extended
to allow users to configure Docker to use the NVIDIA Container Runtime.

For further instructions, see the NVIDIA Container Toolkit [documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)
and specifically the [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Issues and Contributing

[Checkout the Contributing document!](https://github.com/NVIDIA/nvidia-container-toolkit/CONTRIBUTING.md)

* For questions, feature requests, or bugs, [open an issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/new) against the `nvidia-container-toolkit` repository.

----------------------------------------------------------------------------------------------
docker run -d --name=bot --gpus all bot -v /home/dshome/Documents/GeekBrains/CursProjects/NLP/config:/app/config -v /home/dshome/Documents/GeekBrains/CursProjects/NLP/logs:/app/logs bot

```
Tue Dec 26 22:05:20 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-80GB          Off | 00000000:8B:00.0 Off |                    0 |
| N/A   56C    P0             305W / 500W |  50506MiB / 81920MiB |     91%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1061      G   /usr/lib/xorg/Xorg                            4MiB |
|    0   N/A  N/A     13136      C   /usr/bin/python3                          50476MiB |
+---------------------------------------------------------------------------------------+
```