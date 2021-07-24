# ARC 2021 
# Speaker Diarization implement on edge based on transfer learning

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
- Himax WE-I Plus EVB board use UART to communicate with host.

- Host use PySerial to receive data from board.

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
	
# DemoVideo
[Link]
	
