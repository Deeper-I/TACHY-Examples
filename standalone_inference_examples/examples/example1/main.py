import os
import time
import fcntl
import ctypes
import argparse

import numpy as np

from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

REQUIRED = "3.2.2"

try:
    current = version("tachy-rt")
except PackageNotFoundError:
    raise RuntimeError("The tachy-rt package is not installed.")

if Version(current) < Version(REQUIRED):
    raise RuntimeError(
        f"Tachy Runtime >= {REQUIRED} is required (current: {current})."
    )

import tachy_rt.core.functions as rt_core
from tachy_rt.utils import spi_bs_host

APP_BOOT_WAIT_SEC = 100  # NPU boot + application startup time
READY_FLAG_ADDR = 0x20000000
READY_FLAG_POLL_SEC = 3
STATUS_POLL_SEC = 5


def time_wait(sec, show_done=True):
    for _ in range(sec):
        print(".", end="", flush=True)
        time.sleep(1)
    if show_done:
        print(" done!")

def create_args():
    def get_parser():
        """
        Get the parser.
        :return: parser
        """
    
        parser = argparse.ArgumentParser(description='Deep learning example script')
    
        parser.add_argument('--upload_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='true')

        parser.add_argument('--path_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='./tachy-shield')

        args = parser.parse_args()

        args.interface = "spi:host"

        return args
    
    args = get_parser()

    return args

def boot(args):
    if 'spi' not in args.interface or not args.upload_firmware:
        return
    os.system("pinctrl set 4 op dl; sleep 3; pinctrl set 4 op dh")

    ''' Upload firmware to device '''

    data = {
        "spl" : {
            "path" : os.path.join(args.path_firmware, "spl.bin"),
            "addr" : "0x0"
        },
        "uboot" : {
            "path" : os.path.join(args.path_firmware, "u-boot.bin"),
            "addr" : "0x2000_0000"
        },
        "kernel" : {
            "path" : os.path.join(args.path_firmware, "image.ub"),
            "addr" : "0x4000_0000"
        },
        "fpga" : {
            "path" : os.path.join(args.path_firmware, "fpga_top.bin"),
            "addr" : "0x3000_0000"
        }}
    spi_type = args.interface.split(":")[-1]
    ret = rt_core.boot(spi_type, rt_core.DEV_TACHY_SHIELD, data)
    if ret:
        pass
    else:
        print("Failed to boot")
        print("Error code :", rt_core.get_last_error_code())
        exit(-1)

def get_result():
    spi = spi_bs_host.spi
    addr = 0x20000000
    size = 0x100
    data = np.zeros((size), dtype=np.uint8)

    spi_args = spi_bs_host.tachy_bs_spi_data()
    spi_args.addr = addr
    spi_args.size = size
    spi_args.data = data.ctypes.data_as(ctypes.c_void_p)
    while True:
        fcntl.ioctl(spi.fd_data, spi.TACHY_BS_SPI_CMD_READ_SYNC, spi_args)
        tmp = data.tobytes()
        size = tmp[0]
        payload = tmp[4:4 + size]

        bbox = np.frombuffer(tmp[4+size:4+size+4*4], dtype="<f4")[:4].reshape(1,4)[0, :4].astype(int)
        str_ocr = payload.decode("utf-8")

        print("--------- EVENT ---------",
              "\nPLATE :", str_ocr,
              "\nBBOX  :", bbox)

def wait_until_ready(args):
    print("Wait until inference ready...")
    time_wait(APP_BOOT_WAIT_SEC)

    print("Check status")
    # 1) check loaded model status
    while True:
        ret, _dict = rt_core.get_device_status(itf=args.interface)
        if ret:
            if _dict['parameter']['models'] is not None:
                break
            else:
                time.sleep(STATUS_POLL_SEC)

    # 2) check ready flag
    print("Check flag")
    spi = spi_bs_host.spi
    while True:
        value = spi.spi_read(READY_FLAG_ADDR, 4).view(np.uint32)[0]
        if value == 0:
            print(" Inference is ready!")
            break
        time_wait(READY_FLAG_POLL_SEC, show_done=False)
        

if __name__ == '__main__':
    ''' Parse arguments '''
    args = create_args()

    ''' Boot tachy-bs '''
    boot(args)

    ''' Wait until application init done '''
    wait_until_ready(args)

    ''' Get inference result '''
    get_result()
