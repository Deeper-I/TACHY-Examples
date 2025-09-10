[Standalone Inference Mode]
Independent inference mode allows inference to run independently, either without a host or without host control. 
It is primarily used for essential, routine inferences that must run automatically upon system boot. 
While it can secure host resources because it is not under host control, inference cannot be controlled by the host. 
In this mode, inference results are output in stream format and are handled like sensors from the host's perspective. 
This offers excellent scalability and efficiency when configuring large systems.

Basically, the following files are required to run inference:
1. SPI file for BS402 booting
2. U-boot file for BS402
3. FPGA bit files for sensors and communications
4. Host communication driver that receives the inference results
