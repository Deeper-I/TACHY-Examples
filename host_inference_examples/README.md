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
| **Object Detection** | in Street view | - |

> ### Example 1: Object Detection - Street view (YOLOv4 Person/Car/NumberPlate)
This example demonstrates **YOLOv4** running on the Tachy-Shield Edge AI Board.

#### Example1 - requirements
1. **Bootloader file** for BS402 booting
   - [spl](https://gofile.me/5NFjK/MPDyBUKCk)
   - [u-boot](https://gofile.me/5NFjK/HrNppqcEw)
2. **Boot image file** for BS402
   - [image](https://gofile.me/5NFjK/NhuNuKcFe)
3. **FPGA bit files** for sensors and communications
   - [FPGA bit](https://gofile.me/5NFjK/5abA7L1Cf)
4. **Driver files**  see [Required Files](#required-files)
5. **tachyrt file** Required TachyRT model for running the inference example
   - [tachyrt](https://gofile.me/5NFjK/8pjoLs9Ss)
6. **Main executable**
   - Run the example application with:
   ```bash
   python3 main.py \
      --model_path "./model_160x288x3_inv-f.tachyrt" \
      --path_firmware "./tachy-shield" \
      --post_config_path "./post_configs.json"
   ```
   - `--model_path` : path to the YOLOv4 compiled model (.tachyrt)
   - `--path_firmware` : directory containing the TACHY-BS Shield firmware binaries
   - `--post_config_path` : path to the post-processing configuration JSON file

#### Quick start

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Deeper-I/TACHY-Examples
   cd /media/hdd1/sandbox/git/deeper-i/TACHY-Examples/host_inference_examples/examples/example1
   ```
   
2. **Prepare required files**  
   - Download bootloader, boot image, FPGA bit in [here](#example1---requirements)
   - Place **all required files in a single directory (e.g., `./tachy-shield/`)** 
   - This directory will later be passed to `--path_firmware` when running the application  

3. **Build / install drivers**  
   see tachy-rpi-drivers [README.md](https://github.com/Deeper-I/tachy-rpi-drivers/blob/main/README.md)

4. **Run the example application**  
   `python3 main.py --model_path "./model_160x288x3_inv-f.tachyrt" --path_firmware "./tachy-shield" --post_config_path "./post_configs.json"`
