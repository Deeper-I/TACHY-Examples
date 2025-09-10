[Host Inference Mode]
Hosted inference mode control inference on the host (e.g., Raspberry Pi, Banana Pi). 
They are primarily used for solutions requiring inference to occur at specific times or for special postprocessing. 
Since the host processes the inference results from an unprocessed AI model, in most cases, the inference results (transmitted data) increase compared to standalone inference mode.

Basically, the following files are required to run inference:
1. SPI file for BS402 booting
2. U-boot file for BS402
3. FPGA bit files for sensors and communications
4. Main executable for running on the host
