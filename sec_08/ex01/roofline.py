import sys

import pandas as pd
import matplotlib.pyplot as plt


class GPUProp:
    def __init__(self, name, tflops, mem_bw_gBps):
        self.name = name
        self.tflops = tflops
        self.mem_bw_gBps = mem_bw_gBps


class KernelProp:
    def __init__(self, name, flops, bytes):
        self.name = name
        self.flops = flops
        self.bytes = bytes


gpu_props_table = {
    "W6800": GPUProp("W6800", 17.82, 512),
    "MI100": GPUProp("MI100", 23.07, 1229),
    "MI50": GPUProp("MI50", 13.41, 1024),
}


def main():
    ## Two arguments, the first one is a GPU name, and the second one is
    ## an output file from rocprof.
    if len(sys.argv) != 3:
        print("Usage: python roofline.py <GPU name> <rocprof output file>")
        sys.exit(1)

    gpu_name = sys.argv[1]
    rocprof_file = sys.argv[2]

    gpu_prop = getGPUProp(gpu_name)
    kernel_props = getKernelProps(rocprof_file)

    plotRoofline(gpu_prop, kernel_props)


def getGPUProp(gpu_name):
    if gpu_name not in gpu_props_table:
        print("Error: GPU name {} is not supported.".format(gpu_name))
        sys.exit(1)

    gpu_props = gpu_props_table[gpu_name]

    return gpu_props


def getKernelProps(rocprof_file):
    df = pd.read_csv(rocprof_file, header=0, sep=",")

    # CSV file must have the following columns:
    #   KernelName,VALIInsts,FetchSize,WriteSize
    if "KernelName" not in df.columns:
        print("Error: CSV file does not have KernelName column.")
        sys.exit(1)

    if "VALIInsts" not in df.columns:
        print("Error: CSV file does not have VALIInsts column.")
        sys.exit(1)

    if "FetchSize" not in df.columns:
        print("Error: CSV file does not have FetchSize column.")
        sys.exit(1)

    if "WriteSize" not in df.columns:
        print("Error: CSV file does not have WriteSize column.")
        sys.exit(1)

    kernel_props = []
    for i in range(len(df)):
        row = df.iloc[i]
        kernel_name = row["KernelName"]
        v_alu_insts = row["VALIInsts"]
        flops = v_alu_insts * 2

        fetch_size = row["FetchSize"]
        write_size = row["WriteSize"]
        bytes = (fetch_size + write_size) * 1024

        kernel_props.append(KernelProp(kernel_name, flops, bytes))

    return kernel_props


def plotRoofline(gpu_prop, kernel_props):
    plt.figure(figsize=(8, 6))

    # Plot the roofline
    plt.plot(
        [0, gpu_prop.tflops],
        [0, gpu_prop.mem_bw_gBps],
        color="black",
        linestyle="dashed",
        linewidth=1,
    )

    # Save the plot
    plt.savefig("roofline.pdf", bbox_inches="tight")
