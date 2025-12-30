# Standalone Inference Mode

Standalone inference mode allows inference to run **independently**, either without a host device or without host control.  
It is primarily used for **essential, routine inferences** that must start automatically when the system boots.  

Since the process is not controlled by the host, it helps to free up host resources. However, the host cannot directly control the inference once it is running.  

In this mode, inference results are output in a **stream format**, and from the host’s perspective, they are handled similarly to sensor data.  
This approach offers excellent **scalability and efficiency** when configuring large systems.

---

## Required Files
Requires bootloader, image, FPGA bitstream, API packages, drivers, and other resources.
**Note:** The required files may differ depending on the example. Download links are provided in each **Example** section.

In general, the following files are required to run inference in standalone mode:
1. **Bootloader**
   - Required to initialize and boot the Tachy-Shield.
2. **System image**
   - Required to opertate the Tachy-Shield.
3. **FPGA bitstream**
   - Required to configure the FPGA logic for the Tachy-Shield operation.
4. **Drivers**
   - [tachy-rpi-drivers](https://github.com/Deeper-I/tachy-rpi-drivers)
   Required to use the Tachy-Shield device on Raspberry Pi, including the host interface driver and the dummy V4L2 sensor driver.
5. **Tachy Runtime API**
   - A python API package provides the runtime library required to execute inference on Tachy-Shield
6. **Main executable** 
   - for running on the host

---

## Example

| Example | Description | Notes |
|---------|-------------|-------|
| **ANPR** | Automatic Number Plate Recognition | example1 |
| **Object Detection** | Detects person | TODO |
---

> ### Example 1: ANPR (Automatic Number Plate Recogition)
This example demonstrates **Automatic Number Plate Recognition (ANPR)** running on the Tachy-Shield Edge AI Board.
The current implementation is optimized for **Korean license plates** only, and may not work correctly with number plates from other countries.
Use this example if you want to test end-to-end inference including detection and recognition of vehicle license plates.

#### Example1 - requirements
1. **Bootloader**
   - [spl](https://gofile.me/5NFjK/4iUNTeqqF)
   - [u-boot](https://gofile.me/5NFjK/dRkyxLi1d)
2. **System image**
   - [image](https://gofile.me/5NFjK/IaG2NKYGT)
3. **FPGA bitstream**
   - [FPGA bit](https://gofile.me/5NFjK/574JXkL1R)
4. **Drivers**
   - see [Required Files](#required-files)
5. **Tachy Runtime API**
   - [python wheel file](https://gofile.me/5NFjK/9a16ln5LV)
6. **Main executable**
   - Run the example application with:
   ```bash
   python3 main.py \
      --path_firmware "./tachy-shield"
   ```
   - `--path_firmware` : directory containing the Tachy-Shield firmware binaries
