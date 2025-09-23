# Standalone Inference Mode

Standalone inference mode allows inference to run **independently**, either without a host device or without host control.  
It is primarily used for **essential, routine inferences** that must start automatically when the system boots.  

Since the process is not controlled by the host, it helps to free up host resources. However, the host cannot directly control the inference once it is running.  

In this mode, inference results are output in a **stream format**, and from the host’s perspective, they are handled similarly to sensor data.  
This approach offers excellent **scalability and efficiency** when configuring large systems.

---

## Required Files
Requires bootloader, boot image, FPGA bit files, and drivers.  
However, the exact files may vary depending on the example.  
Please refer to the download links provided in each **Example** section below.

In general, the following files are required to run inference in standalone mode:
1. **Bootloader file** for BS402 booting
2. **Boot image file** for BS402
3. **FPGA bit files** for sensors and communications
4. **Driver files**
   - [tachy-rpi-drivers](https://github.com/Deeper-I/tachy-rpi-drivers)
   Drivers required to use the Tachy-Shield device on Raspberry Pi, including the host interface driver and the dummy V4L2 sensor driver.
5. **Main executable** for running on the host

---

## Example

| Example | Description | Notes |
|---------|-------------|-------|
| **ANPR** | Automatic Number Plate Recognition | TODO |
| **Object Detection** | Detects person | TODO |


> ### Example 1: ANPR (Automatic Number Plate Recogition)
This example demonstrates **Automatic Number Plate Recognition (ANPR)** running on the Tachy-Shield Edge AI Board.
The current implementation is optimized for **Korean license plates** only, and may not work correctly with number plates from other countries.
Use this example if you want to test end-to-end inference including detection and recognition of vehicle license plates.

#### requirements
1. **Bootloader file** for BS402 booting
   - [spl](https://gofile.me/5NFjK/MPDyBUKCk)
   - [u-boot](https://gofile.me/5NFjK/HrNppqcEw)
2. **Boot image file** for BS402
3. **FPGA bit files** for sensors and communications
   - [FPGA bit](https://gofile.me/5NFjK/5abA7L1Cf)
4. **Driver files**  see [Required Files](#required-files)
5. **Main executable** 