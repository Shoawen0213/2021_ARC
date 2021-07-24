# ARC Speaker Diarization implement on edge based on transfer learning

Our project is using wei board to do some audio data preprocessing (e.g. down sampling, normalize, data format transform ... ) and then using machine learning model to recognise different speaker and transform audio to txt file.

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
Our project is using wei board to do some audio data preprocessing (e.g. down sampling, normalize, data format transform ... ) and then using machine learning model to recognise different speaker and transform audio to txt file.

* Main Task Of Our Platform
    - microphone
    - Data preprocessing(e.g. down sampling, normalize, data format transform)

## System-Architecture
Himax WE-I Plus EVB board use UART to communicate with host.

host use pyserial to receive data from board.

## Hardware-And-Software-Setup
### Required-Hardware
Himax WE-I Plus EVB board

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
	* using "make" to compile the code (main.c)
	* under windows environment
* img file
	* using "make flash" to generate the img file (output_gnu.img)
	* under linux environment
### Programing
* flash image into board
	* using XMODEM to transmit img file
	* UART bard rate 115200bps
### Geting-Start
* normal mode
	* push reset button on the board
	* execute "Code/Python/demo_A" Jupyter file to receive data from board (recording_a.txt)
	* execute "Code/Python/TRANS_WAV" Jupyter file to transform data to wav file (Recording_0724.wav)
	* execute "Code/Python/Inference" Jupyter file to recognise the wav file 
	
* change voice mode
	* push reset button on the board
	* execute "Code/Python/demo_D" Jupyter file to receive data from board (recording_CV.txt)
	* execute "Code/Python/Chang_voice" Jupyter file to transform data to wav file (Recording_change_voice_0724.wav)
	* execute "Code/Python/Inference" Jupyter file to recognise the wav file
	
# DemoVideo
[Link]
	
