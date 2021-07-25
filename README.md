![](https://github.com/Shoawen0213/2021_ARC/blob/main/doc/synopsys_arc.png)  
# Synopsys ARC 2021 AIoT Design Contest
# OASIS_X-FMR @ NCTU 
## Speaker Diarization implement on edge based on transfer learning

In this contest, we aim to develop a robust, general and accurate speaker diarization model which support configurable parameters.The computation complexity and time-consuming tranmissive time our proposed would be solved by doing data pre-processing with the help of Himax WE-I. 
Our project is using WE-I board to do some audio data preprocessing (e.g. down sampling, normalize, data format transform ... ) and then using machine learning model to recognize different speaker and transform audio into txt file.

* [Introduction](#Introduction)
	* [System Architecture](#System-Architecture)
* [Hardware and Software Setup](#Hardware-And-Software-Setup)
	* [Required Hardware](#Required-Hardware)
	* [Required Software](#Required-Software)
* [User Manual](#user-manual)
    * [Building-image](#Building-image)
    * [Programing](#Programing)
    * [Geting-Start](#Geting-Start)
         * [DemoVideo](#DemoVideo)

# Introduction
we aim to develop a robust, general and accurate speaker diarization model which can be use in different scenario, such as Multi-Speaker Diarization for meeting, contract tracing and video speaker recognition ...etc.  
Our main goal is：
- Simplify the complex procedure and port an inference model on the limited computation platform.
- Make the model more robust and general which can accomplish speaker diarization accurately.
- Provide a convenient way to configure the model.

![image](https://user-images.githubusercontent.com/63163334/126901343-28682ea9-8b5b-4e17-8824-e8098fd529c7.png)


* Main Task of Our Platform
    - Microphone
    - Data preprocessing (e.g. down sampling, normalize, data format transform)

## System-Architecture
![](https://github.com/Shoawen0213/2021_ARC/blob/main/doc/system_ar.png)  
![](https://github.com/Shoawen0213/2021_ARC/blob/main/doc/SAF.png)  
- Himax WE-I Plus EVB board use UART to communicate with host.

- Host use PySerial to receive data from board.

## Hardware-And-Software-Setup
### Required-Hardware
- Himax WE-I Plus EVB board

![](https://github.com/Shoawen0213/2021_ARC/blob/main/doc/17256-Himax_WE-I_Plus_EVB_Endpoint_AI_Development_Board-01.jpg)  

### Required-Software
* ARC_GNU Environment 
* Python 3.6
* UART communication related Package
	* pyserial
* MachineLearning related Package
    * Jupyter (optional)
	* Numpy
	* torch 1.8.1
	* speech_recognition

# User-Manual
### Building-image
* Makefile
	* Folder："Code/Build_img"
	* Using "make" to compile the code (main.c)
	* Under windows environment
* img file
	* Folder："Code/Build_img"
	* Using "make flash" to generate the img file (output_gnu.img)
	* Under linux environment
### Programing
* flash image into board
	* Using XMODEM to transmit img file (output_gnu.img)
	* UART bard rate 115200bps
### Geting-Start
* Normal mode
	* Push reset button on the board
	* Execute "Code/Python/demo_A" Jupyter file to receive data from board       (e.g. recording_a.txt)
	* Execute "Code/Python/TRANS_WAV" Jupyter file to transform data to wav file (e.g. Recording_0724.wav)
	* Execute "Code/Python/Inference" Jupyter file to recognise the wav file 
	
* Voice-Changing mode
	* Push reset button on the board
	* Execute "Code/Python/demo_D" Jupyter file to receive data from board         (e.g. recording_CV.txt)
	* Execute "Code/Python/Chang_voice" Jupyter file to transform data to wav file (e.g. Recording_change_voice_0724.wav)
	* Execute "Code/Python/Inference" Jupyter file to recognise the wav file

# Demo 
![](https://github.com/Shoawen0213/2021_ARC/blob/main/doc/S__56868870.jpg)
# DemoVideo
[Link](https://youtu.be/Y50rdMTMpoo)
	
