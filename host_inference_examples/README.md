# Host Inference Mode

Hosted inference mode controls inference on the **host device** (e.g., Raspberry Pi, Banana Pi).  
It is primarily used for solutions that require inference to occur at specific times or for special post-processing.

Since the host processes the inference results from an unprocessed AI model, in most cases the inference results (transmitted data volume) is **larger** compared to standalone inference mode.

---

## Required Files
Running inference requires boot files, drivers, API packages, and other resources.
However, the exact files may vary depending on the example.  
Please refer to the download links provided in each [Example](#example) section below.

The following files are required to run inference in host-driven mode:

1. **Bootloader file** for BS402 booting
   - [spl](https://gofile.me/5NFjK/MPDyBUKCk)
   - [u-boot](https://gofile.me/5NFjK/HrNppqcEw)
2. **Boot image file** for BS402
   - [image.ub](https://gofile.me/5NFjK/5abA7L1Cf)
3. **FPGA bit files** for sensors and communications
   - [FPGA bit](https://gofile.me/5NFjK/5abA7L1Cf)
4. **Driver files**
   - [tachy-rpi-drivers](https://github.com/Deeper-I/tachy-rpi-drivers)
   Drivers required to use the Tachy-Shield device on Raspberry Pi, including the host interface driver and the dummy V4L2 sensor driver.

5. **Tachy Runtime API**  
   - [tachy_rt.whl](https://gofile.me/5NFjK/yOUZ56BIm) (Python API package provides the runtime library required to execute inference on Tachy-Shield)
6. **Main executable** for running on the host
   - **Note:** The executable differs by example. Download links are provided in each **Example** section.
7. **tachyrt** 
   - A compiled file generated from a trained model, required for execution on the Tachy-Shield NPU.

---

## Example
| Example | Description | Notes |
|---------|-------------|-------|
| **Object Detection** | Detects person | -|
> ### Object Detection (Yolov9 Person)

