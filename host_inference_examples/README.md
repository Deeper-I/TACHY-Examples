# Host Inference Mode

Hosted inference mode controls inference on the **host device** (e.g., Raspberry Pi, Banana Pi).  
It is primarily used for solutions that require inference to occur at specific times or for special post-processing.

Since the host processes the inference results from an unprocessed AI model, in most cases the inference results (transmitted data volume) is **larger** compared to standalone inference mode.

---

## Required Files
Running inference requires boot files, drivers, API packages, and other resources.
**Note:** The required files may differ depending on the example. Download links are provided in each **Example** section.

The following files are required to run inference in host-driven mode:
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
   - The host-side application used to control the device and run inference.
7. **TachyRT model file (`.tachyrt`)** 
   - A compiled model file generated from a trained model, required for execution on the Tachy-Shield NPU.

---

## Example
| Example | Description | Notes |
|---------|-------------|-------|
| **Object Detection** | In street view | example1 |

> ### Example 1: Object Detection - Street view (YOLOv4 Person/Car/NumberPlate)
This example demonstrates **YOLOv4** running on the Tachy-Shield Edge AI Board.

#### Example1 - requirements
1. **Bootloader**
   - [spl](https://gofile.me/5NFjK/4iUNTeqqF)
   - [u-boot](https://gofile.me/5NFjK/dRkyxLi1d)
2. **System image**
   - [image](https://gofile.me/5NFjK/CA5F0acpY)
3. **FPGA bitstream**
   - [FPGA bit](https://gofile.me/5NFjK/574JXkL1R)
4. **Drivers**
   - see [Required Files](#required-files)
5. **Tachy Runtime API**
   - [python wheel file](https://gofile.me/5NFjK/9a16ln5LV)
6. **TachyRT model file (`.tachyrt`)** 
   - [tachyrt](https://gofile.me/5NFjK/8pjoLs9Ss)
7. **Main executable**
   - Run the example application with:
   ```bash
   python3 main.py \
      --model_path "./model_160x288x3_inv-f.tachyrt" \
      --path_firmware "./tachy-shield" \
      --post_config_path "./post_configs.json"
   ```
   - `--model_path` : path to the YOLOv4 compiled model (.tachyrt)
   - `--path_firmware` : directory containing the Tachy-Shield firmware binaries
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
