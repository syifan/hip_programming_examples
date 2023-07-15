import sys

import pandas as pd
import matplotlib.pyplot as plt


class GPUProp:
    def __init__(self, name, freq, tflops, mem_bw_gBps):
        self.name = name
        self.freq = freq
        self.tflops = tflops
        self.mem_bw_gBps = mem_bw_gBps


class KernelProp:
    def __init__(self, name, flops, ai):
        self.name = name
        self.flops = flops
        self.ai = ai


gpu_props_table = {
    "W6800": GPUProp("W6800", 2320, 17.82, 512 / 1024),
    "MI100": GPUProp("MI100", 1502, 23.07, 1229 / 1024),
    "MI50": GPUProp("MI50", 1746, 13.41, 1024 / 1024),
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
    kernel_props = getKernelProps(rocprof_file, gpu_prop)

    plotRoofline(gpu_prop, kernel_props)


def getGPUProp(gpu_name):
    if gpu_name not in gpu_props_table:
        print("Error: GPU name {} is not supported.".format(gpu_name))
        sys.exit(1)

    gpu_props = gpu_props_table[gpu_name]

    return gpu_props


def getKernelProps(rocprof_file, gpu_prop):
    df = pd.read_csv(rocprof_file, header=0, sep=",")

    # CSV file must have the following columns:
    #   KernelName,VALIInsts,FetchSize,WriteSize
    if "KernelName" not in df.columns:
        print("Error: CSV file does not have KernelName column.")
        sys.exit(1)

    if "GRBM_COUNT" not in df.columns:
        print("Error: CSV file does not have GRBM_COUNT column.")
        sys.exit(1)

    if "VALUInsts" not in df.columns:
        print("Error: CSV file does not have VALUInsts column.")
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

        cycles = row["GRBM_COUNT"]
        time = cycles / (gpu_prop.freq * 1000 * 1000)

        v_alu_insts = row["VALUInsts"]
        flop = v_alu_insts * 64 * 2
        flops = flop / time

        fetch_size = row["FetchSize"]
        write_size = row["WriteSize"]
        bytes = fetch_size + write_size
        ai = flop / bytes

        print(
            f"{kernel_name}: cycles-{cycles} cycles, time-{time}s, FLOP-{flop}, FLOPS-{flops}, memory bytes-{bytes}, AI-{ai}"
        )

        kernel_props.append(KernelProp(kernel_name, flops, ai))

    return kernel_props


def plotRoofline(gpu_prop, kernel_props):
    fig, ax = plt.subplots()

    # Plot the roofline
    ax.plot(
        [0, 64],
        [gpu_prop.tflops, gpu_prop.tflops],
        color="black",
        linestyle="dashed",
        linewidth=1,
    )

    ax.plot(
        [0, gpu_prop.tflops / gpu_prop.mem_bw_gBps],
        [0, gpu_prop.tflops],
        color="black",
        linestyle="dashed",
        linewidth=1,
    )

    # Plot the kernels
    for kernel_prop in kernel_props:
        ai = kernel_prop.ai
        flops = kernel_prop.flops
        plt.plot(
            ai,
            flops / (1 << 40),
            marker="o",
            color="black",
            markersize=3,
        )

    ax.set_yscale("log")
    ax.set_xscale("log")

    # Save the plot
    fig.savefig("roofline.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
